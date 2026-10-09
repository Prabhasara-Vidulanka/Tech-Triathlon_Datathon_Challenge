# Datathon Architecture Diagrams

## Task 1: service time and lateness

```mermaid
flowchart LR
    A[Orders and route plans] --> D[Schema and one-to-one join validation]
    B[Outlet, vehicle and district references] --> D
    C[Calendar, traffic and road conditions] --> D
    D --> E[Training-only label construction]
    D --> F[Plan-time feature pipeline]
    E --> G[41-day chronological folds]
    F --> G
    G --> H[GPU XGBoost service regressor]
    G --> I[GPU XGBoost lateness classifier]
    H --> J[Nonnegative service prediction]
    I --> K[Probability audit and bounds]
    J --> L[submission_task1.csv]
    K --> L
```

## Task 2A: weekly demand forecasting

```mermaid
flowchart LR
    A[Training orders] --> C[Deduplicate and combine order history]
    B[Task 1 test orders] --> C
    C --> D[Aggregate by requested ISO week, depot and brand]
    E[Future-known calendar] --> F[Weekly calendar features]
    D --> G[Shifted lags, rolling statistics and trends]
    F --> G
    G --> H[Three rolling 10-week backtests]
    H --> I[Total-volume boosting model]
    H --> J[Fresh chilled-volume boosting model]
    I --> K[Recursive 10-week forecast]
    J --> K
    K --> L[Nonnegative and chilled-total constraints]
    L --> M[submission_task2a.csv]
```

## Task 2B and deployment concept

```mermaid
flowchart LR
    A[Peak-day orders] --> D[Constraint preparation]
    B[Available fleet] --> D
    C[Travel and handling standards] --> D
    D --> E[Exact refrigerated MILP]
    D --> F[Ambient capacity and time packing]
    E --> G[Fairness-first policy comparison]
    F --> G
    G --> H[Official feasibility validator]
    H --> I[submission_task2b.csv and policy]

    J[Saved Task 1 and Task 2A models] --> K[Waypoint planning service]
    K --> L[Dispatcher estimates and capacity planning]
    I -. separate Datathon decision support .-> L
```

The deployment block is a proposed planning interface only. The booklet explicitly states that Datathon integration into the Hackathon application is not required.
