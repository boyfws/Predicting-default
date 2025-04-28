# Intro 

The original dataset can be found [here](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv).  
This dataset was modified; [here](https://www.kaggle.com/code/db0boy/lending-club-loan-data-cleared-preparation) is a Kaggle kernel with the initial preprocessing code (the notebook is identical to the one in this directory).  
The final cleaned dataset is available [here](https://www.kaggle.com/datasets/db0boy/lending-club-loan-data-cleared).  
And [here](https://www.kaggle.com/code/db0boy/credit-scoring-data-processing) the notebook that performs data splitting and adds a few new features


# Features 


1. `funded_amnt` - The total amount committed to that loan at that point in time  
2. `interest_rate` - Interest Rate on the loan  
3. `monthly_payment` - The monthly payment owed by the borrower if the loan originates  
4. `grade` - grade from C1 to A5, from worse to best respectively  
5. `emp_title` - The job title supplied by the Borrower when applying for the loan  
6. `emp_length` - Employment length in years  
7. `home_ownership_status` - The home ownership status provided by the borrower during registration  
8. `annual_income` - The self-reported annual income provided by the borrower during registration  
9. `verification_status` - Indicates if income was verified by LC, not verified, or if the income source was verified  
10. `loan_purpose` - A category provided by the borrower for the loan request  
11. `addr_state` - The state provided by the borrower in the loan application  
12. `dept_paym_income_ratio` - A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income  
13. `num_30+_delinq_in_2yrs` - The number of 30+ days past-due incidences of delinquency  
14. `num_inq_in_6mths` - The number of inquiries in past 6 months (excluding auto and mortgage inquiries)  
15. `mths_since_last_delinq` - The number of months since the borrower's last delinquency  
16. `num_open_credit_lines` - The number of open credit lines in the borrower's credit file  
17. `num_derogatory_pub_rec` - Number of derogatory public records  
18. `total_credit_revolving_bal` - Total credit revolving balance  
19. `used_credit_share` - Revolving line utilization rate, or the amount of credit the borrower is using relative to all assets  
20. `tot_num_credit_lines` - The total number of credit lines currently in the borrower's credit file  
21. `initial_list_status` - The initial listing status of the loan  
22. `remaining_princ_for_tot_amnt_fund` - Remaining outstanding principal for total amount funded  
23. `paym_rec_for_tot_amnt_fund` - Payments received to date for total amount funded  
24. `princ_rec` - Principal received to date  
25. `interest_rec` - Interest received to date  
26. `late_fees_rec` - Late fees received to date  
27. `num_open_trades_in_6mths` - Number of open trades in last 6 months  
28. `num_installment_acc_op_in_12mths` - Number of installment accounts opened in past 12 months  
29. `num_installment_acc_op_in_24mths` - Number of installment accounts opened in past 24 months  
30. `mths_since_last_installment_acc_op` - Months since most recent installment accounts opened  
31. `num_rev_trades_op_in_12mths` - Number of revolving trades opened in past 12 months  
32. `num_rev_trades_op_in_24mths` - Number of revolving trades opened in past 24 months  
33. `max_bal_owed` - Maximum current balance owed on all revolving accounts  
34. `bal_to_cred_lim` - Balance to credit limit on all trades  
35. `num_inq` - Number of personal finance inquiries  
36. `num_inq_in_12mths` - Number of credit inquiries in past 12 months  
37. `mths_since_recent_bankcard_delinq` - Months since most recent bankcard delinquency  
38. `mths_since_recent_revol_delinq` - Months since most recent revolving delinquency  
39. `disbursement_method` - The method by which the borrower receives their loan  
40. `loan_term_months` - Loan duration in months  
41. `issue_date_month` - The month which the loan was funded  
42. `issue_date_year` - The year which the loan was funded  
43. `region_code` - Region code, ranging from “0” in the Northeast (e.g. Massachusetts, Connecticut) to “9” in the West (e.g. California, Alaska)  
44. `earliest_cr_line_month` - The month the borrower's earliest reported credit line was opened  
45. `earliest_cr_line_year` - The year the borrower's earliest reported credit line was opened  

# Added features
Here are the newly added features (engineered features) based on your dataset, described in English:

### **1. Financial Ratio Features**
1. `num_inc_in_fund_amnt`  
   - *Definition*: Ratio of annual income to funded loan amount (`annual_income / funded_amnt`).  
   - *Purpose*: Measures the borrower's income coverage relative to the loan size.  

2. `max_bal_owed_per_income` 
   - *Definition*: Ratio of maximum revolving balance owed to annual income (`max_bal_owed / annual_income`).  
   - *Purpose*: Indicates debt burden relative to income.  

3. `total_credit_revolving_bal_per_income`  
   - *Definition*: Ratio of total revolving credit balance to annual income (`total_credit_revolving_bal / annual_income`).  
   - *Purpose*: Evaluates revolving credit utilization vs. income.  

4. `closed_credit_lines_share`  
   - *Definition*: Proportion of closed credit lines to total credit lines (`(tot_num_credit_lines - num_open_credit_lines) / tot_num_credit_lines`).  
   - *Purpose*: Reflects credit history stability (higher = more closed accounts).  

5. `cost_of_expr`  
   - *Definition*: Annual income divided by employment length (years), with categorical `emp_length` mapped to numerical values.  
   - *Purpose*: Estimates "income per year of experience," proxy for career stability.  



### **2. Date & Cyclical Features**
6. `earliest_cr_line_full`
   - *Definition*: Combined month and year of the earliest credit line (`earliest_cr_line_month + earliest_cr_line_year` as string).  
   - *Purpose*: Creates a unified timestamp feature for credit history age.  

7. `earliest_cr_line_month_sin` / `earliest_cr_line_month_cos` 
   - *Definition*: Trigonometric transformations of the month (`earliest_cr_line_month`) using sine and cosine.  
   - *Purpose*: Encodes cyclicality of months (e.g., Dec and Jan are close in the cycle).  

8. `issue_date_month_num_sin` / `issue_date_month_num_cos`  
   - *Definition*: Same as above but for the loan issue month (`issue_date_month`).  
   - *Purpose*: Captures seasonality effects in loan issuance.  



