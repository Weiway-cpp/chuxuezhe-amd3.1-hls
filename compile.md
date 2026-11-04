Fix the first actionable compiler diagnostic before later cascading errors.
Check the function name, parameter types, headers, declarations and HLS type conversions.
For an undeclared helper, check the supplied public declarations first. If no
implementation is provided, implement the required helper in this source; adding
only a prototype moves the failure to linking. Do not invent private library APIs.
Return complete compilable C++, not a patch, an explanation, or several source files.
Do not change the testbench or hide errors with a replacement main function.
