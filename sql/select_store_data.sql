-- select_store_data.sql
-- 가맹점의 최근 1분기 요일별 매출금액
SELECT 
    day_of_week, 
    SUM(total_sales) AS total_sales
FROM sales_summary
WHERE store_id = %s
  AND summary_date BETWEEN DATE_SUB('2024-06-30', INTERVAL 4 MONTH) AND '2024-06-30'
GROUP BY day_of_week;

-- QUERY_SEPARATOR

-- 가맹점의 최근 1분기 시간대별 매출금액
SELECT 
    SUM(time_00_06) AS sales_00_06,
    SUM(time_06_11) AS sales_06_11,
    SUM(time_11_14) AS sales_11_14,
    SUM(time_14_17) AS sales_14_17,
    SUM(time_17_21) AS sales_17_21,
    SUM(time_21_24) AS sales_21_24
FROM sales_summary
WHERE store_id = %s 
  AND summary_date BETWEEN DATE_SUB('2024-06-30', INTERVAL 4 MONTH) AND '2024-06-30';

-- QUERY_SEPARATOR

-- 가맹점의 최근 1분기 연령대별 매출금액
SELECT 
    SUM(sales_10s) AS sales_10s,
    SUM(sales_20s) AS sales_20s,
    SUM(sales_30s) AS sales_30s,
    SUM(sales_40s) AS sales_40s,
    SUM(sales_50s) AS sales_50s,
    SUM(sales_over_60s) AS sales_over_60s
FROM sales_summary
WHERE store_id = %s 
  AND summary_date BETWEEN DATE_SUB('2024-06-30', INTERVAL 4 MONTH) AND '2024-06-30';

-- QUERY_SEPARATOR

-- 가맹점의 최근 1분기 성별 매출금액
SELECT 
    SUM(male_sales) AS male_sales,
    SUM(female_sales) AS female_sales
FROM sales_summary
WHERE store_id = %s 
  AND summary_date BETWEEN DATE_SUB('2024-06-30', INTERVAL 4 MONTH) AND '2024-06-30';

-- QUERY_SEPARATOR

-- 가맹점의 최근 8분기 분기별 총 매출금액
SELECT 
    CONCAT(YEAR(summary_date), QUARTER(summary_date)) AS quarter_year_code,
    SUM(total_sales) AS total_sales
FROM sales_summary
WHERE store_id = %s
  AND summary_date <= '2024-06-30'
GROUP BY CONCAT(YEAR(summary_date), QUARTER(summary_date))
LIMIT 8;