"""Small public-interface hints; never read a benchmark or reference file."""
import re


def interface_guidance(description, *, testbench=False):
    hints = []
    if re.search(r'\bap_(?:u?fixed)\s*<', description):
        hints.append(
            "For ap_fixed/ap_ufixed, preserve the declared widths, signedness, "
            "quantization and overflow modes at each assignment. Assignment can "
            "round or wrap before the final output; do not assume double arithmetic "
            "or a widened accumulator is equivalent.")
        if testbench:
            hints.append(
                "Derive the fixed-point reference independently using scaled integers "
                "or explicitly modeled quantization at the specified stages. Choose "
                "representable values; do not compare an unquantized floating result "
                "to a quantized output with an arbitrary tolerance. Initialize test "
                "data with small exactly representable integers. Construct constants "
                "in the declared fixed-point type before arithmetic; avoid ambiguous "
                "mixed ap_fixed/double operators in the testbench. Bound inputs by the "
                "declared integer and fractional widths. Derive reference results from "
                "the values actually stored after input quantization, not the original "
                "out-of-range or unquantized literals. Never divide one fixed-point "
                "value by another in the testbench: '/' is ambiguous for these types "
                "and fails to compile. Multiply by an explicit constant, shift, or "
                "convert both operands to double first.")
    if re.search(r'\bhls\s*::\s*stream\s*<', description):
        hints.append(
            "For stream interfaces, derive token counts and framing from the task. "
            "Preserve ordering, sideband fields and termination markers. Do not "
            "invent a token count or use empty() as an end-of-message rule unless "
            "the specification explicitly requires that behavior.")
        if testbench:
            hints.append(
                "Enqueue all required input tokens before calling a blocking DUT; "
                "after it returns, check output availability before each read, "
                "compare every required output and check for unexpected extra tokens. "
                "Do not consume output merely to drain the stream without checks.")
    if re.search(r'\b(?:float|double)\b', description):
        hints.append(
            "Preserve floating-point precision and the specified operation order; "
            "reassociation and fused operations can change observable rounding.")
        if testbench:
            hints.append(
                "Use the task's accuracy requirements. If approximation is allowed, "
                "derive an appropriate absolute-plus-relative error bound; do not "
                "require bit equality for unspecified approximate results. Explicitly "
                "reject unexpected NaNs and infinities: fabs(actual-expected)>tol "
                "alone can silently accept NaN.")
    array_defs = {}
    for element, name, dims in re.findall(
            r'\btypedef\s+([A-Za-z_][\w:]*)\s+([A-Za-z_]\w*)\s*((?:\[\s*\w*\s*\])+)\s*;',
            description):
        array_defs[name] = (element, re.sub(r'\s+', '', dims))
    for name in sorted(array_defs):
        element, dims = array_defs[name]
        if not re.search(r'\b(?:const\s+)?' + re.escape(name) + r'\s*\*', description):
            continue
        parts = re.findall(r'\[[^\]]*\]', dims)
        if len(parts) == 1:
            binding = 'const ' + element + ' *rows = *param;  // rows[i] is element i'
        else:
            binding = (element + ' (*rows)' + ''.join(parts[1:])
                       + ' = *param;  // rows[i][j] is the element')
        hints.append(
            "The interface declares '" + name + "' as an array typedef ('" + dims
            + "') and passes a pointer to it. That parameter points at the whole array, so "
            "'param[i]' is not an element. Begin the function by binding it, replacing the "
            "parameter name: '" + binding + "'. The parentheses are required: "
            "'param[i]' or 'param[i][j]' does not compile against this signature.")
        hints.append(
            "A bare array does not convert to the pointer-to-array parameter '" + name
            + " *'. At every call site declare an object of the typedef itself, "
            "'" + name + " obj;', and pass its address, 'f(..., &obj)'. Do not pass "
            "'uint8_t buf[N]' where '" + name + " *' is expected: the compiler reports "
            "\"no matching function for call\".")
    integer_aliases = sorted({name for name in re.findall(
        r'\btypedef\s+(?:unsigned\s+|signed\s+)?(?:char|short|int|long)\s+([A-Za-z_]\w*)\s*;',
        description)})
    if integer_aliases:
        hints.append(
            "The header defines its own integer names (" + ", ".join(integer_aliases[:8]) +
            "). Use those names or the plain C++ types the header makes available; do not "
            "add <stdint.h> and do not use uint16_t, uint32_t, int8_t or other fixed-width "
            "names unless the header itself declares them.")
    if (re.search(r'\bextract\w*', description, re.I)
            and re.search(r'\bexponent\b', description, re.I)
            and re.search(r'\b(?:mask|masking|bitwise\s+AND)\b', description, re.I)):
        hints.append(
            "The description mentions the stored bias only to explain the encoding; the "
            "requested result is the raw masked field as stored. Do not subtract the bias, "
            "normalize or convert the field to a floating-point value unless the text "
            "explicitly asks for the unbiased or scaled value.")
    entry_points = sorted({name for name in re.findall(
        r'(?m)^\s*(?:extern\s+"C"\s+)?(?:void|int|float|double|bool)\s+([A-Za-z_]\w*)\s*\(',
        description)})
    if len(entry_points) > 1:
        hints.append(
            "The header declares these entry points: " + ", ".join(entry_points[:6]) +
            ". Define every one of them at namespace scope with exactly the declared name and "
            "signature, not only the one named as the top function: a testbench may call any of "
            "them and the build fails with \"undefined symbol\" when one is missing.")
    macros = sorted({name for name in re.findall(
        r'^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(', description, re.M)})
    if macros:
        hints.append(
            "The header already defines " + ", ".join("'" + name + "'" for name in macros[:6])
            + " as a macro (function-like). Do not declare a function, variable or parameter "
            "with any of those names anywhere in your file: the preprocessor rewrites every "
            "occurrence and the result is a syntax error such as \"expected ')'\". Name any "
            "helper differently, for example 'ref_" + macros[0] + "'.")
    return "\n".join(hints)
