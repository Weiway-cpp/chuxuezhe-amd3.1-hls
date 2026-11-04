Preserve the behavior that already passed C simulation.
Replace unsupported runtime allocation or recursion with a bounded equivalent only when task limits justify it.
Avoid unchecked full unrolling or complete array partitioning that can explode hardware size.
Handle dynamic loop bounds according to the specification; do not silently truncate legal inputs.
Use the first synthesis error and relevant source location to guide the repair.
When diagnostics identify DATAFLOW or stream connectivity, check each stream's producer and consumer,
the specified number of reads/writes and end markers, and leftover connections after refactoring.
Keep one writer process and one reader process per stream; do not silence errors by dropping required tokens.
Resolve PIPELINE/DATAFLOW conflicts at the reported hierarchy only. A DATAFLOW region may call pipelined
subfunctions; do not remove legal pipelining from other hierarchy levels or add pragmas speculatively.
