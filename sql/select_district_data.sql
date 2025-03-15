-- 상권
-- 특정 상권의 최근 1분기 요일별 매출금액
SELECT
  quarter_year_code,
    monday_sales_amount AS monday_sales,
    tuesday_sales_amount AS tuesday_sales,
    wednesday_sales_amount AS wednesday_sales,
    thursday_sales_amount AS thursday_sales,
    friday_sales_amount AS friday_sales,
    saturday_sales_amount AS saturday_sales,
    sunday_sales_amount AS sunday_sales
FROM district_statistics
WHERE district_code = %s 
  AND service_category_code = %s
  AND quarter_year_code = CONCAT(YEAR(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)), QUARTER(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)));

-- QUERY_SEPARATOR

-- 특정 상권의 최근 1분기 시간대별 매출금액
SELECT 
    SUM(sales_00_06_amount) AS sales_00_06,
    SUM(sales_06_11_amount) AS sales_06_11,
    SUM(sales_11_14_amount) AS sales_11_14,
    SUM(sales_14_17_amount) AS sales_14_17,
    SUM(sales_17_21_amount) AS sales_17_21,
    SUM(sales_21_24_amount) AS sales_21_24
FROM district_statistics
WHERE district_code = %s 
  AND service_category_code = %s
  AND quarter_year_code = CONCAT(YEAR(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)), QUARTER(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)));

-- QUERY_SEPARATOR

-- 특정 상권의 최근 1분기 연령대별 매출금액
SELECT 
    SUM(age_10_sales_amount) AS sales_10s,
    SUM(age_20_sales_amount) AS sales_20s,
    SUM(age_30_sales_amount) AS sales_30s,
    SUM(age_40_sales_amount) AS sales_40s,
    SUM(age_50_sales_amount) AS sales_50s,
    SUM(age_60_plus_sales_amount) AS sales_60_plus
FROM district_statistics
WHERE district_code = %s 
  AND service_category_code = %s
  AND quarter_year_code = CONCAT(YEAR(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)), QUARTER(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)));

-- QUERY_SEPARATOR

-- 특정 상권의 최근 1분기 성별 매출금액
SELECT 
    SUM(male_sales_amount) AS male_sales,
    SUM(female_sales_amount) AS female_sales
FROM district_statistics
WHERE district_code = %s 
  AND service_category_code = %s
  AND quarter_year_code = CONCAT(YEAR(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)), QUARTER(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)));

-- QUERY_SEPARATOR

-- 특정 상권의 최근 8분기 분기별 총 매출금액
SELECT
  quarter_year_code,
    SUM(monthly_sales_amount) AS total_sales
FROM district_statistics
WHERE district_code = %s 
  AND service_category_code = %s
  AND quarter_year_code BETWEEN '20223' AND '20242'
GROUP BY quarter_year_code;