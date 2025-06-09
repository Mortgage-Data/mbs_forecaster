seller_level_query = f"""
    WITH seller_monthly AS (
        SELECT 
            DATE_TRUNC('month', as_of_month) as month,
            SUM(1) as loan_count,
            SUM(current_investor_loan_upb) as total_upb,
            SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
            SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
            SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
            SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
            CASE WHEN SUM(prepayable_balance) > 0 
                 THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) 
                 ELSE 0 END as cpr,
            AVG(pmms30) as pmms30,
            AVG(pmms30_1m_lag) as pmms30_1m_lag,
            AVG(pmms30_2m_lag) as pmms30_2m_lag,
            ROW_NUMBER() OVER (ORDER BY DATE_TRUNC('month', as_of_month)) - 1 as time_index
        FROM main.gse_sf_mbs a 
        LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
        WHERE is_in_bcpr3 AND prefix = 'CL' AND {seller_where_clause}
        AND as_of_month >= '2022-01-01' AND loan_correction_indicator != 'pri'
        GROUP BY 1
        HAVING loan_count >= 100
    )
    SELECT *,
        LAG(cpr, 1) OVER (ORDER BY month) as cpr_1m_lag,
        LAG(cpr, 3) OVER (ORDER BY month) as cpr_3m_lag,
        AVG(cpr) OVER (ORDER BY month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as cpr_6m_avg,
        weighted_avg_rate - pmms30 as refi_incentive,
        ABS(pmms30 - pmms30_1m_lag) as rate_volatility
    FROM seller_monthly
    ORDER BY month
"""