Use the reported input, expected value and actual value to find the cause.
Trace one counterexample through the supplied code before editing: follow each branch,
shift, accumulator update and assignment, and locate the first intermediate that violates
the public specification. Fix that root cause and necessary dependent expressions while
preserving unrelated working code. Return the complete source, not an edit fragment.
Only an explicitly decimal record receives an added [hex=...] view. Unmarked digits
may already be hexadecimal: preserve their spelling and do not guess their radix.
An added view is an integer representation, not a floating-point numeric conversion.
Derive any bit-field interpretation from the declared interface.
For a model-generated self-test, expected values are fallible. Re-derive the result
from the original specification before editing the design; never hardcode a reported
expected value or change correct specified behavior just to fit a local assertion.
Check the output type's representable range and every partial sum, not only the
width of the testbench reference variable. A 64-bit expected value may be impossible
for a 32-bit output. C++ signed overflow is not implicitly defined as wraparound.
Follow explicit wrap/saturation rules; where overflow is unspecified, exercise
boundary operands in combinations whose intermediates and outputs remain in range.
Do not add casts or hardcoded values solely to make an impossible oracle pass.
Check off-by-one bounds, signed arithmetic, initialization and values that must remain unchanged.
An output mismatch requires a semantic fix; changing only HLS pragmas cannot fix the algorithm.
If tracing exposes a wrong algorithm rather than a local expression, re-derive that
function using a simple invariant and bounded loops. Keep unrelated functions intact.
Distinguish raw packed encodings, unpacked fields, numeric values and intermediate
accumulators at every helper boundary; a correct type alone does not make their roles
interchangeable. Do not repair different input ranges with guessed additive constants.
Recheck zero-length and full-length cases without hardcoding the observed tests.
If the public specification uses encoded floating-point bits, numeric casts are not
bit reinterpretation. Check signed zero, subnormals, infinities, NaN payloads and
rounding only as required by that specification. Do not assume a host float/double
operation reproduces a specified bit-exact result or invent unspecified NaN encodings.
