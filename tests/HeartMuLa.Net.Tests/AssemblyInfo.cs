using Xunit;

// The libtorch RNG is global: tests that seed it (manual_seed) cannot run in parallel with others that use it.
// The suite takes milliseconds except for the real-model tests, which already run serially within the RealModel collection.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
