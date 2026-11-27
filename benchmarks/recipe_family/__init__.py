"""Family-based benchmark for extraction recipes (issue #67 / EX-09).

Synthetic families of repeated layouts (same vendor/template, different values),
including template-drift documents. Provider calls are simulated with a
configurable per-call latency, so call counts and accuracy are exact while wall
time reflects only the simulated latency. Real cost and accuracy on production
families must be measured by the host with its own providers.
"""
