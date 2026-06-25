# SwissTable-style hashmap in pure Julia. The SIMD group probe replaces Base Dict's scalar
# per-slot scan with a Vec{16,UInt8} group compare (one SIMD instruction per 16 control bytes).
# No StrictMode/TypeContracts dep (package rule). SIMD.jl only.
module SwissDict

using SIMD: Vec, vload, bitmask

export SwissDict

# ── constants (mirror Base Dict) ─────────────────────────────────────────────
const _MAXALLOWEDPROBE = 16
const _MAXPROBESHIFT   = 6
const _G               = 16          # group width (Vec{16,UInt8} == SSE2 / NEON lane)
const _EMPTY           = 0x00
const _TOMBSTONE       = 0x7f

# ── struct ───────────────────────────────────────────────────────────────────
# `slots` is length `sz + _G`; indices 1:sz are the real control bytes; indices
# sz+1:sz+_G mirror indices 1:_G so that a group load at any 1-based position is safe.
mutable struct SwissDict{K,V} <: AbstractDict{K,V}
    slots::Memory{UInt8}    # length sz + _G (trailing _G bytes = mirror of first _G)
    keys::Memory{K}         # length sz
    vals::Memory{V}         # length sz
    ndel::Int               # tombstone count
    count::Int              # live entry count
    age::UInt               # monotone version (mirrors Base for detect-concurrent-write)
    idxfloor::Int           # lower bound on the first live slot index
    maxprobe::Int           # worst-case probe distance seen so far

    function SwissDict{K,V}() where {K,V}
        slots = Memory{UInt8}(undef, _G)   # sz=0 base + _G mirror; _G >= 16 minimum
        fill!(slots, _EMPTY)
        new(slots, Memory{K}(undef, 0), Memory{V}(undef, 0), 0, 0, 0, 1, 0)
    end

    function SwissDict{K,V}(d::SwissDict{K,V}) where {K,V}
        new(copy(d.slots), copy(d.keys), copy(d.vals),
            d.ndel, d.count, d.age, d.idxfloor, d.maxprobe)
    end
end

# ── constructors ─────────────────────────────────────────────────────────────
function SwissDict{K,V}(kv) where {K,V}
    h = SwissDict{K,V}()
    Base.haslength(kv) && sizehint!(h, Int(length(kv))::Int)
    for (k, v) in kv
        h[k] = v
    end
    return h
end

SwissDict{K,V}(p::Pair) where {K,V} = setindex!(SwissDict{K,V}(), p.second, p.first)

function SwissDict{K,V}(ps::Pair...) where {K,V}
    h = SwissDict{K,V}()
    sizehint!(h, length(ps))
    for p in ps
        h[p.first] = p.second
    end
    return h
end

SwissDict() = SwissDict{Any,Any}()
SwissDict(kv::Tuple{}) = SwissDict()
Base.copy(d::SwissDict) = SwissDict(d)

SwissDict(ps::Pair{K,V}...) where {K,V} = SwissDict{K,V}(ps)
SwissDict(ps::Pair...)                  = SwissDict(ps)
SwissDict(kv)                           = Base.dict_with_eltype((K, V) -> SwissDict{K, V}, kv, eltype(kv))

Base.empty(::SwissDict, ::Type{K}, ::Type{V}) where {K,V} = SwissDict{K,V}()

# ── internal: table size (next power of two ≥ 16) ────────────────────────────
@inline _tablesz(x::Int) = x < 16 ? 16 : (1 << (8 * sizeof(Int) - leading_zeros(x - 1)))

# ── internal: hash → (index, shorthash7) ─────────────────────────────────────
@inline function _shorthash7(hsh::UInt)
    (hsh >> (8 * sizeof(UInt) - 7)) % UInt8 | 0x80
end

@inline function _hashindex(key, sz::Int)
    hsh = hash(key)::UInt
    idx = ((hsh % Int) & (sz - 1)) + 1
    return idx, _shorthash7(hsh)
end

# ── slot state predicates ─────────────────────────────────────────────────────
@inline _isslotempty(h::SwissDict, i::Int)   = @inbounds h.slots[i] == _EMPTY
@inline _isslotfilled(h::SwissDict, i::Int)  = @inbounds (h.slots[i] & 0x80) != 0
@inline _isslotmissing(h::SwissDict, i::Int) = @inbounds h.slots[i] == _TOMBSTONE

# ── internal: mirror the first _G control bytes into slots[sz+1:sz+_G] ───────
# Call after any write to slots[i] where i ≤ _G, or after rehash.
@inline function _update_mirror!(slots::Memory{UInt8}, sz::Int)
    @inbounds for i in 1:_G
        slots[sz + i] = slots[i]
    end
    nothing
end

# ── SIMD group probe (the one novel piece) ────────────────────────────────────
#
# Scans `slots` in 16-wide groups. `bitmask(g == Vec{16,UInt8}(sh))` is a single
# pcmpeqb+pmovmskb pair on x86 — one instruction per 16 slots.
#
# `slots_ptr` must point to slots[1] (caller: GC.@preserve + pointer(h.slots)).
# `sz` is a power of two ≥ 16. The _G-byte mirror appended to `slots` makes every
# 16-wide group load safe even when `index` is near the end of the table.
#
# Termination: the first group containing an EMPTY byte proves the key absent —
# SwissTable invariant: between the hash position and any stored key, all slots are
# non-empty. No maxprobe early-exit needed (EMPTY detection is exact).
#
# Returns:
#   > 0  → found at that 1-based index
#   -1   → absent
function _find_slot(slots_ptr::Ptr{UInt8}, keys::Memory{K},
                    key::K, sz::Int, sh::UInt8, index::Int) where {K}
    mask  = sz - 1
    vsh   = Vec{_G, UInt8}(sh)
    vzero = Vec{_G, UInt8}(_EMPTY)
    @inbounds while true
        # Load _G control bytes at 1-based `index` (0-based pointer offset = index-1).
        g = vload(Vec{_G, UInt8}, slots_ptr + (index - 1))

        # Check for h2 matches — pcmpeqb + pmovmskb on x86.
        m = bitmask(g == vsh)
        while m != 0x0000
            j = index + trailing_zeros(m)          # logical position (may exceed sz)
            j_real = ((j - 1) & mask) + 1          # wrap into [1, sz]
            k = keys[j_real]
            (key === k || isequal(key, k)) && return j_real
            m &= m - 0x0001                        # clear lowest set bit
        end

        # Any EMPTY in the group ⇒ key is absent.
        bitmask(g == vzero) != 0x0000 && return -1

        # Advance by one group (16-stride), wrapping.
        index = ((index - 1 + _G) & mask) + 1
    end
    # unreachable
end

# ── Lookup (read-only) ────────────────────────────────────────────────────────
function _ht_keyindex(h::SwissDict{K,V}, key) where {K,V}
    h.count == 0 && return -1
    sz = length(h.keys)
    index, sh = _hashindex(key, sz)
    GC.@preserve h begin
        slots_ptr = pointer(h.slots)
        return _find_slot(slots_ptr, h.keys, key, sz, sh, index)
    end
end

# ── Insert-side probe with tombstone reuse ────────────────────────────────────
# Returns (slot, sh) where:
#   slot > 0  → existing key found at `slot` (update in place)
#   slot < 0  → insert at `-slot`
function _ht_keyindex2!(h::SwissDict{K,V}, key::K) where {K,V}
    sz = length(h.keys)
    if sz == 0
        _rehash!(h, 4)
        sz = length(h.keys)
        index, sh = _hashindex(key, sz)
        return -index, sh
    end

    index, sh = _hashindex(key, sz)
    avail     = 0          # first tombstone seen (-slot) or 0
    iter      = 0
    mask      = sz - 1
    keys      = h.keys
    slots     = h.slots

    @inbounds while true
        s = slots[index]
        if s == _EMPTY
            return (avail != 0 ? avail : -index), sh
        end

        if s == _TOMBSTONE
            avail == 0 && (avail = -index)
        elseif s == sh
            k = keys[index]
            (key === k || isequal(key, k)) && return index, sh
        end

        index = (index & mask) + 1
        iter += 1
        iter > h.maxprobe && break
    end

    avail != 0 && return avail, sh

    # Continue scanning for existing key or a free slot (mirroring Base logic).
    maxallowed = max(_MAXALLOWEDPROBE, sz >> _MAXPROBESHIFT)
    @inbounds while iter < maxallowed
        s = slots[index]
        if s != _EMPTY && s != _TOMBSTONE
            if s == sh
                k = keys[index]
                (key === k || isequal(key, k)) && return index, sh
            end
        else
            h.maxprobe = iter
            return -index, sh
        end
        index = (index & mask) + 1
        iter += 1
    end

    _rehash!(h, h.count > 64000 ? sz * 2 : sz * 4)
    return _ht_keyindex2!(h, key)
end

# ── rehash! ───────────────────────────────────────────────────────────────────
function _rehash!(h::SwissDict{K,V}, newsz::Int = length(h.keys)) where {K,V}
    olds  = h.slots
    oldk  = h.keys
    oldv  = h.vals
    sz    = length(oldk)
    newsz = _tablesz(newsz)
    h.age += 1
    h.idxfloor = 1

    if h.count == 0
        slots = Memory{UInt8}(undef, newsz + _G)
        fill!(slots, _EMPTY)
        h.slots   = slots
        h.keys    = Memory{K}(undef, newsz)
        h.vals    = Memory{V}(undef, newsz)
        h.ndel    = 0
        h.maxprobe = 0
        return h
    end

    slots = Memory{UInt8}(undef, newsz + _G)
    fill!(slots, _EMPTY)
    keys = Memory{K}(undef, newsz)
    vals = Memory{V}(undef, newsz)
    age0 = h.age
    count    = 0
    maxprobe = 0
    mask     = newsz - 1

    for i in 1:sz
        @inbounds if (olds[i] & 0x80) != 0
            k  = oldk[i]
            v  = oldv[i]
            index, sh = _hashindex(k, newsz)
            index0 = index
            while slots[index] != _EMPTY
                index = (index & mask) + 1
            end
            probe = (index - index0) & mask
            probe > maxprobe && (maxprobe = probe)
            slots[index] = olds[i]
            keys[index]  = k
            vals[index]  = v
            count += 1
        end
    end

    @assert h.age == age0 "Concurrent writes to SwissDict detected!"
    h.age     += 1
    h.slots    = slots
    h.keys     = keys
    h.vals     = vals
    h.count    = count
    h.ndel     = 0
    h.maxprobe = maxprobe
    _update_mirror!(slots, newsz)
    return h
end

# ── _setindex! (internal: write to a known empty/tombstone slot) ──────────────
@inline function _setindex!(h::SwissDict, v, key, index::Int, sh::UInt8)
    @inbounds begin
        h.ndel -= _isslotmissing(h, index)
        h.slots[index] = sh
        h.keys[index]  = key
        h.vals[index]  = v
        h.count += 1
        h.age   += 1
        index < h.idxfloor && (h.idxfloor = index)
        # maintain mirror if we wrote into the first _G positions
        index <= _G && _update_mirror!(h.slots, length(h.keys))

        sz = length(h.keys)
        if (h.count + h.ndel) * 3 > sz * 2
            _rehash!(h, h.count > 64000 ? h.count * 2 : max(h.count * 4, 4))
        end
    end
    nothing
end

# ── AbstractDict interface ────────────────────────────────────────────────────

Base.length(h::SwissDict) = h.count
Base.isempty(h::SwissDict) = h.count == 0

function Base.sizehint!(d::SwissDict{T}, newsz::Int; shrink::Bool = true) where {T}
    oldsz = length(d.keys)
    newsz = min(max(newsz, length(d)), Base.max_values(T)::Int)
    newsz = _tablesz(cld(3 * newsz, 2))
    return (shrink ? newsz == oldsz : newsz <= oldsz) ? d : _rehash!(d, newsz)
end

function Base.setindex!(h::SwissDict{K,V}, v0, key0) where {K,V}
    key = key0 isa K ? key0 : convert(K, key0)::K
    if !(key0 isa K) && !isequal(key, key0)
        throw(Base.KeyTypeError(K, key0))
    end
    setindex!(h, v0, key)
end

function Base.setindex!(h::SwissDict{K,V}, v0, key0::K) where {K,V}
    v = v0 isa V ? v0 : convert(V, v0)::V
    index, sh = _ht_keyindex2!(h, key0)
    if index > 0
        h.age += 1
        @inbounds h.keys[index] = key0
        @inbounds h.vals[index] = v
    else
        @inbounds _setindex!(h, v, key0, -index, sh)
    end
    return h
end

function Base.getindex(h::SwissDict{K,V}, key) where {K,V}
    index = _ht_keyindex(h, key)
    index < 0 && throw(KeyError(key))
    @inbounds return h.vals[index]::V
end

function Base.get(h::SwissDict{K,V}, key, default) where {K,V}
    index = _ht_keyindex(h, key)
    @inbounds return index < 0 ? default : h.vals[index]::V
end

function Base.get(default::Base.Callable, h::SwissDict{K,V}, key) where {K,V}
    index = _ht_keyindex(h, key)
    @inbounds return index < 0 ? default() : h.vals[index]::V
end

function Base.get!(h::SwissDict{K,V}, key0, default) where {K,V}
    key = key0 isa K ? key0 : convert(K, key0)::K
    index, sh = _ht_keyindex2!(h, key)
    index > 0 && return @inbounds h.vals[index]::V
    v = default isa V ? default : convert(V, default)::V
    @inbounds _setindex!(h, v, key, -index, sh)
    return v
end

function Base.get!(default::Base.Callable, h::SwissDict{K,V}, key0) where {K,V}
    key = key0 isa K ? key0 : convert(K, key0)::K
    index, sh = _ht_keyindex2!(h, key)
    index > 0 && return @inbounds h.vals[index]::V
    age0 = h.age
    v = default()
    v isa V || (v = convert(V, v)::V)
    if h.age != age0
        index, sh = _ht_keyindex2!(h, key)
    end
    if index > 0
        h.age += 1
        @inbounds h.keys[index] = key
        @inbounds h.vals[index] = v
    else
        @inbounds _setindex!(h, v, key, -index, sh)
    end
    return v
end

Base.haskey(h::SwissDict, key) = _ht_keyindex(h, key) >= 0

function _delete!(h::SwissDict{K,V}, index::Int) where {K,V}
    @inbounds begin
        slots = h.slots
        sz    = length(h.keys)
        Base._unsetindex!(h.keys, index)
        Base._unsetindex!(h.vals, index)
        # if next slot is empty, no tombstone needed; back-patch contiguous tombstones
        ndel = 1
        nextind = (index & (sz - 1)) + 1
        if slots[nextind] == _EMPTY
            while true
                ndel -= 1
                slots[index] = _EMPTY
                index <= _G && (slots[sz + index] = _EMPTY)  # update mirror
                index = ((index - 2) & (sz - 1)) + 1
                slots[index] == _TOMBSTONE || break
            end
        else
            slots[index] = _TOMBSTONE
            index <= _G && (slots[sz + index] = _TOMBSTONE)  # update mirror
        end
        h.ndel  += ndel
        h.count -= 1
        h.age   += 1
    end
    return h
end

function Base.delete!(h::SwissDict, key)
    index = _ht_keyindex(h, key)
    index > 0 && _delete!(h, index)
    return h
end

function Base.pop!(h::SwissDict, key)
    index = _ht_keyindex(h, key)
    index > 0 || throw(KeyError(key))
    @inbounds v = h.vals[index]
    _delete!(h, index)
    return v
end

function Base.pop!(h::SwissDict, key, default)
    index = _ht_keyindex(h, key)
    index > 0 || return default
    @inbounds v = h.vals[index]
    _delete!(h, index)
    return v
end

function Base.pop!(h::SwissDict)
    isempty(h) && throw(ArgumentError("dict must be non-empty"))
    idx = _skip_deleted_floor!(h)
    @inbounds k = h.keys[idx]
    @inbounds v = h.vals[idx]
    _delete!(h, idx)
    return k => v
end

function Base.empty!(h::SwissDict{K,V}) where {K,V}
    sz = length(h.keys)
    fill!(h.slots, _EMPTY)
    for i in 1:sz
        Base._unsetindex!(h.keys, i)
        Base._unsetindex!(h.vals, i)
    end
    h.ndel     = 0
    h.count    = 0
    h.maxprobe = 0
    h.age     += 1
    h.idxfloor = max(1, sz)
    return h
end

# ── iteration ─────────────────────────────────────────────────────────────────
function _skip_deleted(h::SwissDict, i::Int)
    L = length(h.keys)
    @inbounds while i <= L
        _isslotfilled(h, i) && return i
        i += 1
    end
    return 0
end

function _skip_deleted_floor!(h::SwissDict)
    idx = _skip_deleted(h, h.idxfloor)
    idx != 0 && (h.idxfloor = idx)
    return idx
end

@inline function _iterate(t::SwissDict{K,V}, i::Int) where {K,V}
    i == 0 && return nothing
    @inbounds return (Pair{K,V}(t.keys[i], t.vals[i]), i == typemax(Int) ? 0 : i + 1)
end

Base.iterate(t::SwissDict) = _iterate(t, _skip_deleted(t, t.idxfloor))
Base.iterate(t::SwissDict, i::Int) = _iterate(t, _skip_deleted(t, i))

# keys/values iterators (Base provides defaults via iterate, but the floor trick speeds them up)
function Base.iterate(v::Base.KeySet{<:Any, <:SwissDict}, i::Int = v.dict.idxfloor)
    i == 0 && return nothing
    i = _skip_deleted(v.dict, i)
    i == 0 && return nothing
    @inbounds return (v.dict.keys[i], i == typemax(Int) ? 0 : i + 1)
end

function Base.iterate(v::Base.ValueIterator{<:SwissDict}, i::Int = v.dict.idxfloor)
    i == 0 && return nothing
    i = _skip_deleted(v.dict, i)
    i == 0 && return nothing
    @inbounds return (v.dict.vals[i], i == typemax(Int) ? 0 : i + 1)
end

end # module SwissDict
