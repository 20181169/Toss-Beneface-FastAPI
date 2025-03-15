SELECT
    service_category_name, similar_stores, closed_stores, opened_stores, franchise_stores,
    municipal_name, administrative_name, district_type_name, district_change_index,
    total_floating_population, male_floating_population, female_floating_population,
    quarter_year_code, age_10_floating_population, age_20_floating_population, age_30_floating_population,
    age_40_floating_population, age_50_floating_population, age_60_plus_floating_population,
    floating_population_00_06, floating_population_06_11, floating_population_11_14, floating_population_14_17,
    floating_population_17_21, floating_population_21_24, floating_population_monday, floating_population_tuesday,
    floating_population_wednesday, floating_population_thursday, floating_population_friday, floating_population_saturday,
    floating_population_sunday
FROM district_statistics
WHERE district_code = %s
  AND service_category_code = %s
  AND quarter_year_code = CONCAT(YEAR(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)), QUARTER(DATE_SUB(CURDATE(), INTERVAL 12 MONTH)));