-- Business questions for the Telco churn dataset.
-- Every query is preceded by a "-- name: <name>" line; src/analysis.py splits
-- the file on those markers and saves each result to reports/tables/<name>.csv.

-- name: overview
-- Headline numbers: customer base, churn, and the monthly revenue churn puts at risk.
SELECT
    COUNT(*)                                                              AS customers,
    SUM(churn_flag)                                                       AS churned_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)                          AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2)                                         AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS churned_monthly_revenue,
    ROUND(
        100.0 * SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END)
        / SUM(MonthlyCharges), 2
    )                                                                     AS churned_revenue_share_pct
FROM customers;

-- name: churn_by_contract
-- Does contract commitment relate to churn?
SELECT
    Contract,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2)                AS monthly_revenue
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure
-- Churn by life-cycle stage: first year, second year, and long-tenured customers.
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '13-24'
        WHEN tenure <= 48 THEN '25-48'
        ELSE '49-72'
    END                                          AS tenure_group,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                AS avg_monthly_charges
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- name: churn_by_payment_method
-- Which payment methods show the highest churn?
SELECT
    PaymentMethod,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- name: churn_by_internet_service
-- Does the internet product line matter?
SELECT
    InternetService,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2)                AS monthly_revenue
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tech_support
-- Churn with and without support/security add-ons, among internet customers only.
SELECT
    'TechSupport'                                AS support_feature,
    TechSupport                                  AS support_level,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE InternetService <> 'No'
GROUP BY TechSupport
UNION ALL
SELECT
    'OnlineSecurity'                             AS support_feature,
    OnlineSecurity                               AS support_level,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE InternetService <> 'No'
GROUP BY OnlineSecurity
ORDER BY support_feature, churn_rate_pct DESC;

-- name: contract_by_tenure
-- Contract type crossed with life-cycle stage, built on a CTE.
WITH customer_stage AS (
    SELECT
        Contract,
        tenure,
        churn_flag,
        CASE
            WHEN tenure <= 12 THEN '0-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49-72'
        END AS tenure_group
    FROM customers
)
SELECT
    Contract,
    tenure_group,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customer_stage
GROUP BY Contract, tenure_group
ORDER BY Contract, MIN(tenure);

-- name: segment_ranking
-- Contract x internet segments ranked by churned monthly revenue, with each
-- segment's share of all churned revenue (window functions, no self-join).
WITH segments AS (
    SELECT
        Contract,
        InternetService,
        COUNT(*)                                                          AS customers,
        SUM(churn_flag)                                                   AS churned,
        ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS churned_monthly_revenue
    FROM customers
    GROUP BY Contract, InternetService
)
SELECT
    Contract,
    InternetService,
    customers,
    churned,
    churned_monthly_revenue,
    ROUND(
        100.0 * churned_monthly_revenue / SUM(churned_monthly_revenue) OVER (),
        2
    )                                                  AS share_of_churned_revenue_pct,
    RANK() OVER (ORDER BY churned_monthly_revenue DESC) AS revenue_rank
FROM segments
ORDER BY revenue_rank;

-- name: high_risk_segment
-- The highest-risk combination: month-to-month, fiber optic, first year.
WITH segment AS (
    SELECT
        COUNT(*)                                                               AS customers,
        SUM(churn_flag)                                                        AS churned,
        ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)                           AS churn_rate_pct,
        ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS churned_monthly_revenue,
        ROUND(AVG(MonthlyCharges), 2)                                          AS avg_monthly_charges
    FROM customers
    WHERE Contract = 'Month-to-month'
      AND InternetService = 'Fiber optic'
      AND tenure <= 12
),
totals AS (
    SELECT ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS total_churned_revenue
    FROM customers
)
SELECT
    segment.*,
    ROUND(100.0 * segment.churned_monthly_revenue / totals.total_churned_revenue, 2) AS share_of_churned_revenue_pct
FROM segment, totals;
