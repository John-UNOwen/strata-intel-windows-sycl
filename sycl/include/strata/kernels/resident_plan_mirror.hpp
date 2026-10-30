// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
// The mirror lane travels per window, not per process: resident_plan (see kernels/verify_kernels.hpp) takes this
// layer's slice of the pinned-host address table (`mir`, 0 = none) and routing counts (`use`, optional) computed
// from the window's VerifyHits.  A layer-split stage therefore passes its own stage's table - the table and the
// residency slice are allocated on the same card, so its kernel reads only its own context's device memory.
// #1980's fault (a later stage slicing the first stage's table and reading another allocation, xe page fault on the
// second card) is now impossible: there is no global window to fall outside.
}
