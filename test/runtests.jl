# Entry point: ReTestItems discovers and runs every `@testitem` under test/. Each crate's tests are
# tagged so a single crate can be triggered in isolation, e.g.:
#   julia --project -e 'using ReTestItems, BlazingPorts; runtests(BlazingPorts; tags=[:smallmatrix])'
using ReTestItems
using BlazingPorts

runtests(BlazingPorts)
