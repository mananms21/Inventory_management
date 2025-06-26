USE urban_inventory;

-- INVENTORY OVERVIEW
-- Inventory by Store, Region, Category
SELECT
    i.Store_ID,
    s.Region,
    p.Category,
    SUM(i.Inventory_Level) AS total_inventory
FROM inventory_facts i
JOIN stores s ON i.Store_ID = s.Store_ID
JOIN products p ON i.Product_ID = p.Product_ID
GROUP BY i.Store_ID, s.Region, p.Category
ORDER BY total_inventory DESC;

-- Count of SKUs
SELECT COUNT(DISTINCT product_id) AS total_unique_skus
FROM urban_inventory.products;
-- Count of Stockouts
SELECT 
    COUNT(*) AS understocked_count
FROM inventory_facts
WHERE Inventory_Level  < Demand_Forecast;

SELECT 
    COUNT(*) AS understocked_count
FROM inventory_facts
WHERE Inventory_Level  < Units_Sold;

-- Count of Overstocked Items: (Assuming overstock means inventory > forecast * 1.5)
SELECT 
    COUNT(*) AS overstocked_count
FROM inventory_facts
WHERE Inventory_Level > (Demand_Forecast * 1.5);

-- SALES PERFORMANCE
-- Units Sold per Product, Store, Category, Region
SELECT 
    i.Product_ID,
    i.Store_ID,
    p.Category,
    s.Region,
    SUM(i.Units_Sold) AS total_units_sold
FROM inventory_facts i
JOIN products p ON i.Product_ID = p.Product_ID
JOIN stores s ON i.Store_ID = s.Store_ID
GROUP BY i.Product_ID, i.Store_ID, p.Category, s.Region
ORDER BY total_units_sold DESC;

-- Revenue per SKU
SELECT 
    Product_ID,
    SUM(Units_Sold * Price) AS revenue
FROM inventory_facts
GROUP BY Product_ID
ORDER BY revenue DESC;

-- Most Sold Products in Last 30 Days
SELECT 
    Product_ID,
    SUM(Units_Sold) AS total_sold_last_30_days
FROM inventory_facts
WHERE Date >= (
    SELECT MAX(Date) FROM inventory_facts
) - INTERVAL 30 DAY
GROUP BY Product_ID
ORDER BY total_sold_last_30_days DESC;

-- Least Sold Products in Last 30 Days
SELECT 
    Product_ID,
    SUM(Units_Sold) AS total_sold_last_30_days
FROM inventory_facts
WHERE Date >= (
    SELECT MAX(Date) FROM inventory_facts
) - INTERVAL 30 DAY
GROUP BY Product_ID
ORDER BY total_sold_last_30_days ASC;

-- Average Discount by Category
SELECT 
    p.Category,
    ROUND(AVG(i.Discount), 2) AS avg_discount
FROM inventory_facts i
JOIN products p ON i.Product_ID = p.Product_ID
GROUP BY p.Category
ORDER BY avg_discount DESC;

-- Average Discount by Store
SELECT 
    i.Store_ID,
    ROUND(AVG(i.Discount), 2) AS avg_discount
FROM inventory_facts i
GROUP BY i.Store_ID
ORDER BY avg_discount DESC;

-- Units Sold per Store & SKU Over Time
SELECT 
    Date,
    Store_ID,
    Product_ID,
    Region,
    SUM(Units_Sold) AS total_units_sold
FROM inventory_data
GROUP BY Date, Store_ID, Region, Product_ID
ORDER BY total_units_sold DESC;

-- ORDER PATTERNS
-- Units Ordered per Store & SKU Over Time
SELECT 
    Date,
    Store_ID,
    Product_ID,
    Region,
    SUM(Units_Ordered) AS total_units_ordered
FROM inventory_data
GROUP BY Date, Store_ID, Region, Product_ID
ORDER BY total_units_ordered DESC;

-- Selling Frequency for Each Product/Store
SELECT 
    Store_ID,
    Region,
    Product_ID,
    COUNT(*) AS sell_days
FROM inventory_data
WHERE Units_Sold > 0
GROUP BY Store_ID, Region, Product_ID
ORDER BY sell_days DESC;

-- Order Frequency for Each Product/Store
SELECT 
    Store_ID,
    Region,
    Product_ID,
    COUNT(*) AS order_days
FROM inventory_data
WHERE Units_Ordered > 0
GROUP BY Store_ID, Region, Product_ID
ORDER BY order_days DESC;

-- Lead Time Between Order and Stock Refill
-- Detect change in inventory after an order
WITH inventory_changes AS (
  SELECT 
    Date,
    Store_ID,
    Region,
    Product_ID,
    Units_Sold,
    Units_Ordered,
    Inventory_Level,
    LAG(Inventory_Level) OVER (PARTITION BY Store_ID, Region, Product_ID ORDER BY Date) AS prev_inventory
  FROM inventory_data
),
inventory_deltas AS (
  SELECT *,
         (Inventory_Level - prev_inventory) AS inventory_diff
  FROM inventory_changes
)
SELECT *
FROM inventory_deltas
WHERE Units_Ordered > 0 ;

-- PROMOTIONS & HOLIDAYS IMPACT
-- Sales Uplift During Promotion/Holiday 
SELECT 
    i.Date,
    e.Holiday_Promotion,
    SUM(i.Units_Sold) AS total_units_sold
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY i.Date, e.Holiday_Promotion
ORDER BY i.Date;

-- Inventory Changes During Promotions 
SELECT 
    e.Holiday_Promotion,
    ROUND(AVG(i.Inventory_Level), 2) AS avg_inventory
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY e.Holiday_Promotion;

-- COMPETITOR PRICE IMPACT
-- Price Difference vs Units Sold
SELECT 
    i.Product_ID,
    ROUND(AVG(i.Price - e.Competitor_Pricing), 2) AS avg_price_gap,
    ROUND(AVG(i.Units_Sold), 2) AS avg_units_sold
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY i.Product_ID
ORDER BY avg_price_gap DESC;

-- Stockout Root Cause
-- Identify frequent stockouts and check if demand forecast was high. (We do not have any stockouts)
SELECT 
    i.Store_ID,
    i.Product_ID,
    COUNT(*) AS stockout_days,
    ROUND(AVG(i.Demand_Forecast), 2) AS avg_forecast
FROM inventory_facts i
WHERE i.Inventory_Level = 0
GROUP BY i.Store_ID, i.Product_ID
ORDER BY stockout_days DESC;

-- Dead Stock Analysis
-- Find SKUs that had no sales but still have inventory - No such inventory
SELECT 
    i.Product_ID,
    COUNT(*) AS no_sales_days,
    ROUND(AVG(i.Inventory_Level), 2) AS avg_inventory
FROM inventory_facts i
WHERE i.Units_Sold = 0 AND i.Inventory_Level > 0
GROUP BY i.Product_ID
HAVING no_sales_days > 10
ORDER BY no_sales_days DESC;

-- Demand vs Inventory Mismatch
-- Compare forecast vs actual to check accuracy.
SELECT 
    Product_ID,
    ROUND(AVG(ABS(Demand_Forecast - Units_Sold)), 2) AS avg_forecast_error
FROM inventory_facts
GROUP BY Product_ID
ORDER BY avg_forecast_error DESC;

-- Regional Performance Variation
-- Measure how weather, promotions, or region affect sales.
SELECT 
    s.Region,
    ROUND(AVG(i.Units_Sold), 2) AS avg_units_sold,
    SUM(CASE WHEN i.Inventory_Level = 0 THEN 1 ELSE 0 END) AS total_stockouts,
    SUM(CASE WHEN i.Inventory_Level > i.Demand_Forecast THEN 1 ELSE 0 END) AS overstocks
FROM inventory_facts i
JOIN stores s ON i.Store_ID = s.Store_ID
GROUP BY s.Region;

-- Impact of promotion and weather
SELECT 
    e.Weather_Condition,
    e.Holiday_Promotion,
    ROUND(AVG(i.Units_Sold), 2) AS avg_units_sold
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY e.Weather_Condition, e.Holiday_Promotion
ORDER BY weather_Condition;
-- SKU Turnover Rate
-- Ratio of Units Sold to Average Inventory.
SELECT 
    Product_ID,
    ROUND(SUM(Units_Sold) / NULLIF(AVG(Inventory_Level), 0), 2) AS turnover_ratio
FROM inventory_facts
GROUP BY Product_ID
ORDER BY turnover_ratio DESC;

-- Dynamic Reorder Thresholds
-- Rolling 7-day average demand per product-store (basic reorder point estimate)
SELECT 
    Store_ID,
    Product_ID,
    ROUND(AVG(Units_Sold), 2) AS avg_daily_sales_7d,
    ROUND(AVG(Units_Sold) * 7, 2) AS reorder_point
FROM inventory_facts
WHERE Date >= (
    SELECT MAX(Date) FROM inventory_facts
) - INTERVAL 7 DAY
GROUP BY Store_ID, Product_ID;

-- Highlight products near reorder threshold:
WITH recent_sales AS (
    SELECT 
        Store_ID,
        Product_ID,
        ROUND(AVG(Units_Sold) * 7, 2) AS reorder_point
    FROM inventory_facts
    WHERE Date >= (
        SELECT MAX(Date) FROM inventory_facts
    ) - INTERVAL 7 DAY
    GROUP BY Store_ID, Product_ID
)
SELECT 
    i.Store_ID,
    i.Product_ID,
    i.Inventory_Level,
    rs.reorder_point
FROM inventory_facts i
JOIN recent_sales rs 
    ON i.Store_ID = rs.Store_ID AND i.Product_ID = rs.Product_ID
WHERE i.Date = (SELECT MAX(Date) FROM inventory_facts)
  AND i.Inventory_Level <= rs.reorder_point;


-- Supplier Performance (Inferred)
-- We can’t track actual suppliers but can infer restocking delay by checking when Units Ordered is followed by no inventory change.
-- Estimate delivery lag: days when Units Ordered > 0 and inventory stays low
SELECT 
    Store_ID,
    Product_ID,
    Date,
    Units_Ordered,
    Inventory_Level,
    LAG(Inventory_Level) OVER (PARTITION BY Store_ID, Product_ID ORDER BY Date) AS prev_inventory
FROM inventory_facts
WHERE Units_Ordered > 0
ORDER BY Store_ID, Product_ID, Date;

-- Price Optimization / Elasticity
-- Measure how sales vary with price levels per product (basic elasticity).
SELECT 
    Product_ID,
    ROUND(Price, 0) AS price_bin,
    AVG(Units_Sold) AS avg_units_sold
FROM inventory_facts
GROUP BY Product_ID, price_bin
ORDER BY Product_ID, price_bin;

-- Find "sweet spot" price per product:
SELECT 
    Product_ID,
    Price,
    AVG(Units_Sold) AS avg_units_sold
FROM inventory_facts
GROUP BY Product_ID, Price
ORDER BY Product_ID, avg_units_sold DESC;

WITH avg_sales_per_price AS (
    SELECT 
        Product_ID,
        Price,
        AVG(Units_Sold) AS avg_units_sold
    FROM inventory_facts
    GROUP BY Product_ID, Price
),
ranked_sales AS (
    SELECT *,
        RANK() OVER (PARTITION BY Product_ID ORDER BY avg_units_sold DESC) AS rnk
    FROM avg_sales_per_price
)
SELECT Product_ID, Price, avg_units_sold
FROM ranked_sales
WHERE rnk = 1;
-- Weather-Driven Demand
-- Correlate weather with units sold per category or product.
SELECT 
    e.Weather_Condition,
    ROUND(AVG(i.Units_Sold), 2) AS avg_units_sold
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY e.Weather_Condition
ORDER BY avg_units_sold DESC;

-- To find avg unit sold of any category Example: Electronics sales in rainy weather:
SELECT 
    e.Weather_Condition,
    ROUND(AVG(i.Units_Sold), 2) AS avg_electronics_sales
FROM inventory_facts i
JOIN products p ON i.Product_ID = p.Product_ID
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
WHERE p.Category = 'Electronics'
GROUP BY e.Weather_Condition
ORDER BY avg_electronics_sales DESC;

-- Promotion Effectiveness
-- Compare sales during promotions vs. non-promotions.
SELECT 
    i.Product_ID,
    e.Holiday_Promotion,
    ROUND(AVG(i.Units_Sold), 2) AS avg_units_sold,
    ROUND(AVG(i.Discount), 2) AS avg_discount
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY i.Product_ID, e.Holiday_Promotion
ORDER BY i.Product_ID, e.Holiday_Promotion;

-- ROI estimate per product:
SELECT 
    i.Product_ID,
    SUM(CASE WHEN e.Holiday_Promotion = 1 THEN i.Units_Sold ELSE 0 END) AS promo_sales,
    SUM(CASE WHEN e.Holiday_Promotion = 0 THEN i.Units_Sold ELSE 0 END) AS non_promo_sales,
    ROUND(SUM(i.Discount * i.Price * i.Units_Sold) / NULLIF(SUM(i.Price * i.Units_Sold), 0), 2) AS promo_cost_ratio
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY i.Product_ID;

-- Store Clustering (Sales Patterns): ➤ Objective: Group stores based on total sales, popular categories, or sales volatility to enable inventory transfer, shared promotions, or regional planning.
-- a) Prepare a sales pattern profile per store and category:
SELECT 
    i.Store_ID,
    p.Category,
    SUM(i.Units_Sold) AS total_units_sold
FROM inventory_facts i
JOIN products p ON i.Product_ID = p.Product_ID
GROUP BY i.Store_ID, p.Category
ORDER BY i.Store_ID, total_units_sold DESC;

-- b) Simplified store profile: average & volatility
SELECT 
    Store_ID,
    ROUND(AVG(Units_Sold), 2) AS avg_daily_sales,
    ROUND(STDDEV(Units_Sold), 2) AS sales_volatility
FROM inventory_facts
GROUP BY Store_ID;

-- Seasonal Forecasting for Key SKUs
-- a) Identify top SKUs:
SELECT 
    Product_ID,
    SUM(Units_Sold) AS total_sold
FROM inventory_facts
GROUP BY Product_ID
ORDER BY total_sold DESC
LIMIT 10;

--  b) Monthly trend for each key SKU:
SELECT 
    DATE_FORMAT(Date, '%Y-%m') AS month,
    Product_ID,
    SUM(Units_Sold) AS monthly_sales
FROM inventory_facts
GROUP BY month, Product_ID
ORDER BY Product_ID, month;

-- c) Average sales by seasonality per SKU
SELECT 
    e.Seasonality,
    i.Product_ID,
    ROUND(AVG(i.Units_Sold), 2) AS avg_seasonal_sales
FROM inventory_facts i
JOIN environment_facts e ON i.Date = e.Date AND i.Store_ID = e.Store_ID
GROUP BY e.Seasonality, i.Product_ID
ORDER BY i.Product_ID;

-- Anomaly Detection: Anomalies = sudden spikes or drops in: Units_Sold, Units_Ordered, Inventory_Level
-- a) Detect abnormal drop or spike in sales:
WITH product_sales AS (
  SELECT 
    Date,
    Product_ID,
    Store_ID,
    Units_Sold,
    LAG(Units_Sold) OVER (PARTITION BY Product_ID, Store_ID ORDER BY Date) AS prev_sales
  FROM inventory_facts
)
SELECT 
    Date,
    Product_ID,
    Store_ID,
    Units_Sold,
    prev_sales,
    (Units_Sold - prev_sales) AS delta,
    ROUND((Units_Sold - prev_sales) / NULLIF(prev_sales, 0) * 100, 2) AS pct_change
FROM product_sales
WHERE ABS(Units_Sold - prev_sales) > 20  -- Customize this threshold
ORDER BY ABS(delta) DESC;
-- We can flag: % drop > 80% or % spike > 150% as suspicious

-- b) Inventory Level Gaps (unexpected drops):
WITH inv_changes AS (
  SELECT 
    Date,
    Store_ID,
    Product_ID,
    Inventory_Level,
    LAG(Inventory_Level) OVER (PARTITION BY Store_ID, Product_ID ORDER BY Date) AS prev_inv
  FROM inventory_facts
)
SELECT 
    *,
    (Inventory_Level - prev_inv) AS changes
FROM inv_changes
WHERE ABS(Inventory_Level - prev_inv) > 50  -- Customize this threshold
ORDER BY ABS(changes) DESC;

CREATE TEMPORARY TABLE log_data AS
SELECT
    product_id,
    price,
    AVG(units_sold) AS avg_units,
    LOG(price) AS log_price,
    LOG(NULLIF(AVG(units_sold), 0)) AS log_units
FROM inventory_facts
GROUP BY product_id, price;

CREATE TEMPORARY TABLE log_means AS
SELECT
    product_id,
    AVG(log_price) AS avg_log_price,
    AVG(log_units) AS avg_log_units
FROM log_data
GROUP BY product_id;

CREATE TEMPORARY TABLE elasticity_calc AS
SELECT
    d.product_id,
    SUM((log_price - m.avg_log_price) * (log_units - m.avg_log_units)) AS numerator,
    SUM(POW((log_price - m.avg_log_price), 2)) AS denominator
FROM log_data d
JOIN log_means m ON d.product_id = m.product_id
GROUP BY d.product_id;


CREATE TEMPORARY TABLE price_elasticity AS
SELECT
    product_id,
    CASE 
        WHEN denominator != 0 THEN numerator / denominator
        ELSE 0
    END AS elasticity
FROM elasticity_calc;

CREATE TEMPORARY TABLE avg_units_by_price AS
SELECT
    product_id,
    price,
    AVG(units_sold) AS avg_units_sold
FROM inventory_facts
GROUP BY product_id, price;

CREATE TEMPORARY TABLE sweet_spot AS
SELECT 
    product_id,
    price AS sweet_spot_price
FROM (
    SELECT
        product_id,
        price,
        AVG(units_sold) AS avg_units_sold,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY AVG(units_sold) DESC) AS rn
    FROM inventory_facts
    GROUP BY product_id, price
) t
WHERE rn = 1;

CREATE TEMPORARY TABLE avg_data AS
SELECT
    product_id,
    AVG(price) AS avg_price,
    AVG(units_sold) AS avg_units
FROM inventory_facts
GROUP BY product_id;

SELECT
    s.product_id,
    s.sweet_spot_price,
    ROUND(a.avg_units * POWER(s.sweet_spot_price / a.avg_price, e.elasticity), 2) AS est_units,
    ROUND(s.sweet_spot_price * a.avg_units * POWER(s.sweet_spot_price / a.avg_price, e.elasticity), 2) AS est_revenue
FROM sweet_spot s
JOIN price_elasticity e ON s.product_id = e.product_id
JOIN avg_data a ON s.product_id = a.product_id;


SELECT
    ROUND(SUM(s.sweet_spot_price * a.avg_units * POWER(s.sweet_spot_price / a.avg_price, e.elasticity)), 2) AS total_estimated_revenue
FROM sweet_spot s
JOIN price_elasticity e ON s.product_id = e.product_id
JOIN avg_data a ON s.product_id = a.product_id;


SELECT
    ROUND(SUM(price * units_sold), 2) AS total_actual_revenue
FROM inventory_facts;

CREATE TEMPORARY TABLE adjusted_demand AS
SELECT
    f.product_id,
    f.price AS actual_price,
    f.units_sold AS actual_units,
    s.sweet_spot_price,
    a.avg_price,
    a.avg_units,
    e.elasticity,
    
    -- Estimated units using elasticity formula
    ROUND(a.avg_units * POWER(s.sweet_spot_price / a.avg_price, e.elasticity), 2) AS est_units_at_sweet_price,
    
    -- Revenue if this transaction was priced at sweet spot with estimated units
    ROUND(s.sweet_spot_price * a.avg_units * POWER(s.sweet_spot_price / a.avg_price, e.elasticity), 2) AS est_revenue_for_this_row

FROM inventory_facts f
JOIN sweet_spot s ON f.product_id = s.product_id
JOIN price_elasticity e ON f.product_id = e.product_id
JOIN avg_data a ON f.product_id = a.product_id;

SELECT
    ROUND(SUM(est_revenue_for_this_row), 2) AS total_estimated_revenue_using_sweet_spot
FROM adjusted_demand;
