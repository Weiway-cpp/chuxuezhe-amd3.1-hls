Preserve the exact top-level function signature, array sizes and observable behavior.
Use bounded storage and deterministic initialization. Do not invent inputs or output values.
Respect signedness and integer overflow semantics in the task.
Avoid dynamic allocation, ordinary recursive calls and non-synthesizable I/O in the design.
Start with a compact, straightforward implementation. Unless the public task explicitly
requires a throughput or memory layout, omit ARRAY_PARTITION, UNROLL, DATAFLOW and forced
PIPELINE II directives. In particular, do not completely partition a large array dimension
just to accelerate a simple element loop. Establish functional and synthesis correctness first.
Derive arithmetic from the stated input representation and recurrence, not the algorithm name.
Do not invent a normalization, encoding conversion or nonlinear transform absent from the task.
When immutable interface headers are provided, include them by their given names and reuse their declarations.
Do not redefine their structs, typedefs, or macros, and do not replace or implement a supplied testbench.
Provide definitions for the public non-inline functions declared by those headers,
including initialization or setup functions that a caller may use before the top.
A declaration alone does not supply an implementation. Check each public function
against the task and implement its specified behavior, even when the top does not call it.
Every non-library helper you call must be defined in this source or provided by the
public headers. A familiar function name does not imply a linked implementation.
For bit-encoded values, distinguish bit fields from numeric values. Derive widths,
masks, shifts and signedness from the public interface; avoid shifts by the type width.
