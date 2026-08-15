# T9445 Markdown Pressure SOP

Control token `T9445-MD-PRESSURE-17`.

## First Wafer Gate

- Hold the lot after a pressure alarm.
- Review RF match before release.

| Signal | Limit | Unit |
| --- | ---: | --- |
| Chamber Pressure | 12 | Pa |

```text
release = leak_check_passed and monitor_wafers >= 2
```
