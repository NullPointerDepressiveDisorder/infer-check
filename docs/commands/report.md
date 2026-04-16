# report

Generate a report from previously saved result JSON files. Supports HTML and JSON output formats.

## CLI Reference

::: mkdocs-click
    :module: infer_check.cli
    :command: main
    :prog_name: infer-check
    :subcommand: report
    :style: table
    :show_subcommand_aliases:

## How it works

1. **Scan** -- recursively finds all `.json` files in the results directory.
2. **Load** -- reads each file and collects all result objects. Handles both single objects and arrays. Skips files that fail to parse.
3. **Generate** -- delegates to the format-specific exporter (HTML or JSON).
4. **Open** -- for HTML reports, automatically opens the report in your default browser.

## Examples

Generate an HTML report from all results:

```bash
infer-check report ./results/ --format html
```

Generate a JSON report to a specific file:

```bash
infer-check report ./results/ --format json --output ./summary.json
```

Report from a specific command's results:

```bash
infer-check report ./results/compare/ --format html
```

## Notes

- The report command does not have `--max-tokens` or `--num-prompts` options since it operates on previously generated results.
- Result files from any command (sweep, compare, diff, stress, determinism) can be mixed in the same directory. The report handles heterogeneous result types.
- If the HTML reporting module is not available, a minimal HTML page with raw JSON data is generated as a fallback.
