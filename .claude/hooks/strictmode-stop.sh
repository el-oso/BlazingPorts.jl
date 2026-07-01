#!/usr/bin/env bash
# StrictMode Stop hook: audit only when src/ changed this turn; block the stop on failures.
# Template: StrictMode.jl docs/src/agents.md ("Wiring it into a CI or agent loop").
input=$(cat)
grep -q '"stop_hook_active":true' <<<"$input" && exit 0   # loop guard: don't re-block our own stop

cd "$(dirname "$0")/../.." || exit 0
hash=$(find src -name '*.jl' | sort | xargs cat | md5sum | cut -d' ' -f1)
stamp=.claude/hooks/.src-hash
[[ -f $stamp && $(cat "$stamp") == "$hash" ]] && exit 0    # src untouched → free

# ponytail: cold julia ~30-60s once per src-touching turn; DaemonMode daemon if latency hurts.
if ! out=$(julia --project=bench bench/strictmode_audit.jl 2>&1); then
    echo "StrictMode audit failed — fix these findings before stopping:" >&2
    tail -40 <<<"$out" >&2
    exit 2                                                 # blocks the stop, stderr reaches the agent
fi
echo "$hash" > "$stamp"                                    # only stamp a clean audit
