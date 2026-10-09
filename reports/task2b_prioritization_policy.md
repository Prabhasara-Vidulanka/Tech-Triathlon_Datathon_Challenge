# Task 2B Peak-Day Prioritization Policy

## Objective

The allocation first protects outlets skipped yesterday, then gives priority to stores with the longest service gap. Within that fairness rule, chilled Fresh orders receive preference because a festival is one week away and chilled stock cannot use ambient vehicles. Remaining capacity is used to maximize coverage without consuming a reefer or van when a standard ambient truck can do the job.

## Bottlenecks and calculations

Only 28 of 38 listed vehicles are available. The binding resource is the four-vehicle reefer fleet: three reefer trucks and one reefer van, for eight trip slots. Chilled Fresh demand spans seven districts. The three chilled van-only stops total 6.013 m³ but 1,095.7 kg, above the reefer van's 1,040 kg one-trip limit, so the only reefer van must make two trips. That leaves six reefer-truck trips for the non-van chilled districts.

Each trip uses the published calculation:

```text
outbound district minutes
+ inter-stop minutes × (orders - 1)
+ brand/dock service allowances
```

Fresh minutes remain below 270 for every vehicle; Style and Tech minutes remain below 480. The final plan uses 29 trips across 15 vehicles, including all eight reefer trips and three van trips. The official validator passes every rule.

## Result

- 77 of 85 orders are served, totaling 313.829 m³ and 55,705.7 kg.
- All 10 orders deferred yesterday are served.
- 68 Fresh orders and 19 of 26 chilled orders are served.
- Five of six van-only orders are served.
- Eight orders are deferred: seven chilled Fresh orders and one Style order.

The Style order `S1-078` is individually impossible: its 40.660 m³ volume exceeds the 38.0 m³ capacity of the largest compatible vehicle. The maximum-coverage solve proves that at least six orders must be deferred. A count-only plan could serve 79 orders but would defer one order that was already skipped yesterday. The selected plan deliberately accepts two additional low-priority deferrals to protect every prior deferral and increase the served sum of `days_since_last_served` from 128 to 130.

The seven chilled deferrals are caused by reefer trip scarcity and district purity, not unused ambient capacity. They are lower-priority choices after protecting the previously deferred Gampaha and Puttalam chilled orders. Operationally, these deferrals risk Fresh stockouts and missed festival-ramp sales, so they should be placed first in the next refrigerated dispatch. `S1-078` requires a larger vehicle, a split-order exception, or supplier rescheduling; none is legal in this scenario.
