import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorerpython3
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, jarque_bera
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import yule_walker
# Be name Khoda

homeValueIndex = pd.read_csv("Zillow Home Value Index (ZHVI.csv")
observedRentIndex = pd.read_csv("Zillow Observed Rent Index (ZORI).csv")
unemployment = pd.read_csv("Colorado_unemployment.csv")
new_construction_home_sold = pd.read_csv("New_construction_Home)sold.csv")
new_private_housing_unit_authorized = pd.read_csv("New Private Housing Units Authorized by Building Permits 1-Unit Structures for Colorado (COBP1FHSA).csv")
market_yield_on_10year_treasury = pd.read_csv("Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10).csv")
labor_force_participation_rate = pd.read_csv("Labor Force Participation Rate for Colorado (LBSSA08).csv")
heatIndex = pd.read_csv("HeatIndex.csv")
forSale_inventory = pd.read_csv("for_sale_inventory.csv")
consumer_sentiment_index = pd.read_csv("Consumer_sentiment.csv")
cpi = pd.read_excel("Consumer Price Index for All Urban Consumers (CPI-U).xlsx")
average_hourly_earning_private = pd.read_csv("Average Hourly Earnings of All Employees Total Private in Colorado (SMU08000000500000003).csv")
mortgage_rate_30y = pd.read_csv("30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US).csv")


#New data
affordable_HomePrice_20_down = pd.read_csv("affordableHomePrice_20Percent_down_Smooth.csv")
median_daySToClose = pd.read_csv("medianDaystoCloseMonthly.csv")
median_daysToPending = pd.read_csv("medianDaysToPendingRaw.csv")
median_list_price = pd.read_csv("medianListPriceRaw.csv")
median_PriceCut = pd.read_csv("medianPriceCutSFR_Raw.csv")
median_SalesPrice = pd.read_csv("medianSalesPriceRawSFR.csv")
median_SalesToList_Ration = pd.read_csv("MedianSaleTolistRatio_Raw.csv")
mortgage_payment_20Down = pd.read_csv("mortgagepaymentat20Percent.csv")
new_contruction_Median_SalesPrice = pd.read_csv("newConstructionMedianSalePrice_SFR_Raw.csv")
new_construction_MedianSalesPrice_perSQF = pd.read_csv("newConstructionMedianSalePricePerSQF_RAW_SFR.csv")
new_homeOwnerAffordibility_20Down = pd.read_csv("newHomeOwnerAffordibility_20Percent_down_Smooth.csv")
new_homeOwnerIncomeNeeded_20Down = pd.read_csv("newHomeOwnerIncomeNeeded_20_percent_smooth.csv")
new_listing = pd.read_csv("newListingRaw.csv")
new_PendingListing = pd.read_csv("newPendingListingRAW.csv")
new_Renter_affordibility = pd.read_csv("newRenterAffordebility_smooth.csv")
new_Renter_incomeNeeded = pd.read_csv("newRenterIncomeNeeded_Smooth.csv")
perccent_Of_homeSold_Above = pd.read_csv("percentOfHomesSoldAbove.csv")
perccent_Of_homeSold_Below = pd.read_csv("percentOfHomesSoldBelow.csv")
sales_count_nowCast = pd.read_csv("salesCountnowCastRAW.csv")
share__of_listing_withPriceCut = pd.read_csv("shareOfListingWithPriceCutSFR_RAW.csv")
total_transaction_value = pd.read_csv("totaalTransactionValueRAWSFROnly.csv")
total_mortgage_payment_20Down = pd.read_csv("totalMortgagePaymentat20Percent.csv")
yearToSave_20Down = pd.read_csv("yearsToSave_20_percent_down_smooth.csv")


#Cleaning New Data

affordable_HomePrice_20_down = affordable_HomePrice_20_down[affordable_HomePrice_20_down["StateName"] == "CO"]
median_daySToClose = median_daySToClose[median_daySToClose["StateName"] == "CO"]
median_daysToPending = median_daysToPending[median_daysToPending["StateName"] == "CO"]
median_list_price = median_list_price[median_list_price["StateName"] == "CO"]
median_PriceCut = median_PriceCut[median_PriceCut["StateName"] == "CO"]
median_SalesPrice = median_SalesPrice[median_SalesPrice["StateName"] == "CO"]
median_SalesToList_Ration = median_SalesToList_Ration[median_SalesToList_Ration["StateName"] == "CO"]
mortgage_payment_20Down = mortgage_payment_20Down[mortgage_payment_20Down["StateName"] == "CO"]
new_contruction_Median_SalesPrice = new_contruction_Median_SalesPrice[new_contruction_Median_SalesPrice["StateName"] == "CO" ]
new_construction_MedianSalesPrice_perSQF = new_construction_MedianSalesPrice_perSQF[new_construction_MedianSalesPrice_perSQF["StateName"] == "CO"]
new_homeOwnerAffordibility_20Down = new_homeOwnerAffordibility_20Down[new_homeOwnerAffordibility_20Down["StateName"] == "CO"]
new_homeOwnerIncomeNeeded_20Down = new_homeOwnerIncomeNeeded_20Down[new_homeOwnerIncomeNeeded_20Down["StateName"] == "CO"]
new_listing = new_listing[new_listing["StateName"] == "CO"]
new_PendingListing = new_PendingListing[new_PendingListing["StateName"] == "CO"]
new_Renter_affordibility = new_Renter_affordibility[new_Renter_affordibility["StateName"] == "CO"]
new_Renter_incomeNeeded = new_Renter_incomeNeeded[new_Renter_incomeNeeded["StateName"] == "CO"]
perccent_Of_homeSold_Above = perccent_Of_homeSold_Above[perccent_Of_homeSold_Above["StateName"] == "CO"]
perccent_Of_homeSold_Below = perccent_Of_homeSold_Below[perccent_Of_homeSold_Below["StateName"] == "CO"]
sales_count_nowCast = sales_count_nowCast[sales_count_nowCast["StateName"] == "CO"]
share__of_listing_withPriceCut = share__of_listing_withPriceCut[share__of_listing_withPriceCut["StateName"] == "CO"]
total_transaction_value = total_transaction_value[total_transaction_value["StateName"] == "CO"]
total_mortgage_payment_20Down = total_mortgage_payment_20Down[total_mortgage_payment_20Down["StateName"] == "CO"]
yearToSave_20Down = yearToSave_20Down[yearToSave_20Down["StateName"] == "CO"]
affordable_HomePrice_20_down["RegionName"].unique

def getDenver(df):
    df = df[df["RegionName"].str.strip() == "Denver, CO"]
    df = df.drop(columns=["RegionID", "SizeRank", "RegionType", "StateName"])
    df.columns = [df.columns[0]] + list(pd.to_datetime(df.columns[1:], errors='coerce'))
    return df


affordable_HomePrice_20_down = getDenver(affordable_HomePrice_20_down)
affordable_HomePrice_20_down.columns
median_daySToClose = getDenver(median_daySToClose)
median_daySToClose.columns
median_daysToPending = getDenver(median_daysToPending)
median_list_price = getDenver(median_list_price)
median_PriceCut = getDenver(median_PriceCut)
median_SalesPrice = getDenver(median_SalesPrice)
median_SalesToList_Ration = getDenver(median_SalesToList_Ration)
mortgage_payment_20Down = getDenver(mortgage_payment_20Down)
new_contruction_Median_SalesPrice = getDenver(new_contruction_Median_SalesPrice)
new_construction_MedianSalesPrice_perSQF = getDenver(new_construction_MedianSalesPrice_perSQF)
new_homeOwnerAffordibility_20Down = getDenver(new_homeOwnerAffordibility_20Down)
new_homeOwnerIncomeNeeded_20Down = getDenver(new_homeOwnerIncomeNeeded_20Down)
new_listing = getDenver(new_listing)
new_PendingListing = getDenver(new_PendingListing)
new_Renter_affordibility = getDenver(new_Renter_affordibility)
new_Renter_incomeNeeded = getDenver(new_Renter_incomeNeeded)
perccent_Of_homeSold_Above = getDenver(perccent_Of_homeSold_Above)
perccent_Of_homeSold_Below = getDenver(perccent_Of_homeSold_Below)
sales_count_nowCast = getDenver(sales_count_nowCast)
share__of_listing_withPriceCut = getDenver(share__of_listing_withPriceCut)
total_transaction_value = getDenver(total_transaction_value)
total_mortgage_payment_20Down = getDenver(total_mortgage_payment_20Down)
yearToSave_20Down = getDenver(yearToSave_20Down)

yearToSave_20Down.isna().sum()
#Selecting the desired date 
start = pd.Timestamp("2018-04-30")
end = pd.Timestamp("2025-01-31")

def getDate(df,start,end):
    df = df[["RegionName"] + [col for col in df.columns if isinstance(col, pd.Timestamp) and start <= col <= end]]
    return df

def cleaning_regionNames(df):
    df = df.copy()
    df["RegionName"] = df["RegionName"].str.replace(", CO", "", regex=False).str.strip().str.lower()
    return df

affordable_HomePrice_20_down = getDate(affordable_HomePrice_20_down,start=start, end=end)
affordable_HomePrice_20_down = cleaning_regionNames(affordable_HomePrice_20_down)
median_daySToClose = getDate(median_daySToClose,start=start,end=end)
median_daySToClose = cleaning_regionNames(median_daySToClose)
median_daysToPending = getDate(median_daysToPending,start=start, end=end)
median_daysToPending = cleaning_regionNames(median_daysToPending)
median_list_price = getDate(median_list_price,start=start,end=end)
median_list_price = cleaning_regionNames(median_list_price)
median_SalesPrice = getDate(median_SalesPrice,start=start,end=end)
median_SalesPrice = cleaning_regionNames(median_SalesPrice)
median_SalesToList_Ration = getDate(median_SalesToList_Ration,start=start,end=end)
median_SalesToList_Ration = cleaning_regionNames(median_SalesToList_Ration)
mortgage_payment_20Down = getDate(mortgage_payment_20Down,start=start,end=end)
mortgage_payment_20Down = cleaning_regionNames(mortgage_payment_20Down)
new_contruction_Median_SalesPrice = getDate(new_contruction_Median_SalesPrice,start=start,end=end)
new_contruction_Median_SalesPrice = cleaning_regionNames(new_contruction_Median_SalesPrice)
new_construction_MedianSalesPrice_perSQF = getDate(new_construction_MedianSalesPrice_perSQF,start=start,end=end)
new_construction_MedianSalesPrice_perSQF = cleaning_regionNames(new_construction_MedianSalesPrice_perSQF)
new_homeOwnerAffordibility_20Down = getDate(new_homeOwnerAffordibility_20Down,start=start,end=end)
new_homeOwnerAffordibility_20Down = cleaning_regionNames(new_homeOwnerAffordibility_20Down)
new_homeOwnerIncomeNeeded_20Down = getDate(new_homeOwnerIncomeNeeded_20Down,start=start,end=end)
new_homeOwnerIncomeNeeded_20Down = cleaning_regionNames(new_homeOwnerIncomeNeeded_20Down)
new_listing = getDate(new_listing,start=start,end=end)
new_listing = cleaning_regionNames(new_listing)
new_PendingListing = getDate(new_PendingListing,start=start,end=end)
new_PendingListing = cleaning_regionNames(new_PendingListing)
new_Renter_affordibility = getDate(new_Renter_affordibility,start=start,end=end)
new_Renter_affordibility = cleaning_regionNames(new_Renter_affordibility)
new_Renter_incomeNeeded = getDate(new_Renter_incomeNeeded,start=start,end=end)
new_Renter_incomeNeeded = cleaning_regionNames(new_Renter_incomeNeeded)
perccent_Of_homeSold_Above = getDate(perccent_Of_homeSold_Above,start=start,end=end)
perccent_Of_homeSold_Above = cleaning_regionNames(perccent_Of_homeSold_Above)
perccent_Of_homeSold_Below = getDate(perccent_Of_homeSold_Below,start=start, end=end)
perccent_Of_homeSold_Below = cleaning_regionNames(perccent_Of_homeSold_Below)
sales_count_nowCast = getDate(sales_count_nowCast,start=start,end=end)
sales_count_nowCast = cleaning_regionNames(sales_count_nowCast)
share__of_listing_withPriceCut = getDate(share__of_listing_withPriceCut,start=start,end=end)
share__of_listing_withPriceCut = cleaning_regionNames(share__of_listing_withPriceCut)
total_transaction_value = getDate(total_transaction_value,start=start,end=end)
total_transaction_value = cleaning_regionNames(total_transaction_value)
total_mortgage_payment_20Down = getDate(total_mortgage_payment_20Down,start=start,end=end)
total_mortgage_payment_20Down = cleaning_regionNames(total_mortgage_payment_20Down)
yearToSave_20Down = getDate(yearToSave_20Down,start=start,end=end)
yearToSave_20Down = cleaning_regionNames(yearToSave_20Down)
def melt_zillow(df, name):
    df = df.copy()
    df.columns = [str(col).strip() if not isinstance(col, pd.Timestamp) else col for col in df.columns]
    df_melted = df.melt(id_vars ="RegionName", var_name = 'date', value_name = name)
    df_melted["date"] = pd.to_datetime(df_melted['date'], errors='coerce')
    return df_melted


affordable_HomePrice_20_down_melted = melt_zillow(affordable_HomePrice_20_down,'affordable_HomePrice_20_down')
median_daySToClose_melted = melt_zillow(median_daySToClose,"median_daySToClose_melted")
median_daysToPending_melted = melt_zillow(median_daysToPending,"median_daysToPending")
median_list_price_melted = melt_zillow(median_list_price,"median_list_price")
median_SalesPrice_melted = melt_zillow(median_SalesPrice, "median_sales_price")
median_SalesToList_Ration_melted = melt_zillow(median_SalesToList_Ration,"median_SalesToList_Ration")
mortgage_payment_20Down_melted = melt_zillow(mortgage_payment_20Down, "mortgage_payment_20Down")
new_contruction_Median_SalesPrice_melted = melt_zillow(new_contruction_Median_SalesPrice,"new_contruction_Median_SalesPrice")
new_construction_MedianSalesPrice_perSQF_melted = melt_zillow(new_construction_MedianSalesPrice_perSQF,"new_construction_MedianSalesPrice_perSQF")
new_homeOwnerAffordibility_20Down_melted = melt_zillow(new_homeOwnerAffordibility_20Down,"new_homeOwnerAffordibility_20Down")
new_homeOwnerIncomeNeeded_20Down_melted = melt_zillow(new_homeOwnerIncomeNeeded_20Down,"new_homeOwnerIncomeNeeded_20Down")
new_listing_melted = melt_zillow(new_listing,"new_listings")
new_PendingListing_melted = melt_zillow(new_PendingListing,"new_PendingListing")
new_Renter_affordibility_melted = melt_zillow(new_Renter_affordibility, "new_Renter_affordibility")
new_Renter_incomeNeeded_melted = melt_zillow(new_Renter_incomeNeeded, "new_Renter_incomeNeeded")
perccent_Of_homeSold_Above_melted = melt_zillow(perccent_Of_homeSold_Above, "perccent_Of_homeSold_Above")
perccent_Of_homeSold_Below_melted = melt_zillow(perccent_Of_homeSold_Below,"perccent_Of_homeSold_Below")
sales_count_nowCast_melted = melt_zillow(sales_count_nowCast,"sales_count_nowCast")
share__of_listing_withPriceCut_melted = melt_zillow(share__of_listing_withPriceCut, "share__of_listing_withPriceCut")
total_transaction_value_melted = melt_zillow(total_transaction_value,"total_transaction_value")
total_mortgage_payment_20Down_melted = melt_zillow(total_mortgage_payment_20Down,"total_mortgage_payment_20Down")
yearToSave_20Down_melted = melt_zillow(yearToSave_20Down,"yearToSave_20Down")

affordable_HomePrice_20_down_melted

big_df = affordable_HomePrice_20_down_melted.copy()
big_df
big_df = big_df.merge(median_daySToClose_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(median_daysToPending_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(median_list_price_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(median_SalesPrice_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(median_SalesToList_Ration_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(mortgage_payment_20Down_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_contruction_Median_SalesPrice_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_construction_MedianSalesPrice_perSQF_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_homeOwnerAffordibility_20Down_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_homeOwnerIncomeNeeded_20Down_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_listing_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_PendingListing_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_Renter_affordibility_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(new_Renter_incomeNeeded_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(perccent_Of_homeSold_Above_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(perccent_Of_homeSold_Below_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(sales_count_nowCast_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(share__of_listing_withPriceCut_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(total_transaction_value_melted, on=["date", "RegionName"], how='outer')
big_df = big_df.merge(total_mortgage_payment_20Down_melted,on=["date", "RegionName"], how='outer')
big_df = big_df.merge(yearToSave_20Down_melted,on=["date", "RegionName"], how='outer')
len(big_df.columns)



#Cleainig Region Name




#Cleaning, and Selecting from March 2018 to Feb 2025
homeValueIndex = homeValueIndex[homeValueIndex["State"] == 'CO']





homeValueIndex = homeValueIndex[homeValueIndex["RegionName"].isin(['Denver'])]
homeValueIndex = homeValueIndex.drop(columns=['RegionID', 'SizeRank','RegionType','StateName','State','Metro','CountyName'])
homeValueIndex.columns = pd.to_datetime(homeValueIndex.columns, errors='ignore')
homeValueIndex
homeValueIndex = homeValueIndex.drop(homeValueIndex.iloc[:,1:219], axis=1)
homeValueIndex

observedRentIndex = observedRentIndex[observedRentIndex["StateName"] == 'CO']
observedRentIndex = observedRentIndex[observedRentIndex["RegionName"].isin(['Denver, CO'])]
observedRentIndex = observedRentIndex.drop(columns=['RegionID', 'SizeRank','RegionType','StateName'])
observedRentIndex.columns = pd.to_datetime(observedRentIndex.columns, errors='ignore')
observedRentIndex.iloc[:,1:39]
observedRentIndex = observedRentIndex.drop(observedRentIndex.iloc[:,1:39], axis=1)
observedRentIndex

new_construction_home_sold = new_construction_home_sold[new_construction_home_sold["StateName"] == 'CO']
new_construction_home_sold["RegionName"]
new_construction_home_sold = new_construction_home_sold[new_construction_home_sold["RegionName"].isin(['Denver, CO'])]
new_construction_home_sold
new_construction_home_sold = new_construction_home_sold.drop(columns=['RegionID', 'SizeRank','RegionType','StateName'])
new_construction_home_sold.columns
new_construction_home_sold.columns = pd.to_datetime(new_construction_home_sold.columns, errors='ignore')
new_construction_home_sold.iloc[:,1:3]
new_construction_home_sold = new_construction_home_sold.drop(new_construction_home_sold.iloc[:,1:3], axis=1)
new_construction_home_sold


heatIndex = heatIndex[heatIndex["StateName"] == 'CO']
heatIndex["RegionName"]
heatIndex = heatIndex[heatIndex["RegionName"].isin(['Denver, CO'])]
heatIndex
heatIndex = heatIndex.drop(columns=['RegionID', 'SizeRank','RegionType','StateName'])
heatIndex.columns
heatIndex.columns = pd.to_datetime(heatIndex.columns, errors='ignore')
heatIndex.iloc[:,1:3]
heatIndex = heatIndex.drop(heatIndex.iloc[:,1:3], axis=1)
heatIndex

forSale_inventory = forSale_inventory[forSale_inventory["StateName"] == 'CO']
forSale_inventory["RegionName"]
forSale_inventory = forSale_inventory[forSale_inventory["RegionName"].isin(['Denver, CO'])]
forSale_inventory
forSale_inventory = forSale_inventory.drop(columns=['RegionID', 'SizeRank','RegionType','StateName'])
forSale_inventory.columns
forSale_inventory.columns = pd.to_datetime(forSale_inventory.columns, errors='ignore')
forSale_inventory

unemployment.columns
unemployment["observation_date"] = pd.to_datetime(unemployment["observation_date"], errors='ignore')
unemployment.rename(columns={'COUR':'unemploymentRate'}, inplace=True)

new_private_housing_unit_authorized["observation_date"] = pd.to_datetime(new_private_housing_unit_authorized["observation_date"], errors='ignore')
new_private_housing_unit_authorized.columns
new_private_housing_unit_authorized.rename(columns={'COBP1FHSA': "NewPrivateHousingUnitAuthorized"}, inplace=True)

market_yield_on_10year_treasury["observation_date"] = pd.to_datetime(market_yield_on_10year_treasury["observation_date"], errors='ignore')
market_yield_on_10year_treasury.columns
market_yield_on_10year_treasury.rename(columns={'DGS10':'marketYeild10YTreasury'}, inplace=True)

labor_force_participation_rate["observation_date"] = pd.to_datetime(labor_force_participation_rate["observation_date"], errors='ignore')
labor_force_participation_rate.columns
labor_force_participation_rate.rename(columns={'LBSSA08':'LaborParticipationRate'}, inplace=True)


consumer_sentiment_index["observation_date"] = pd.to_datetime(consumer_sentiment_index["observation_date"], errors='ignore')
consumer_sentiment_index.columns
consumer_sentiment_index.rename(columns={"UMCSENT":"ConsumerSentimentIndex"}, inplace=True)
cpi_xlsx = pd.read_excel("CPi.xlsx")
cpi_xlsx
cpi_df = pd.DataFrame(cpi_xlsx)
cpi_df.columns
cpi_df["Date "] = pd.to_datetime(cpi_df["Date "], errors='ignore')
cpi_df.columns = cpi_df.columns.str.strip()
cpi_df.columns
cpi_df.rename(columns={'value':'cpi'}, inplace=True)

average_hourly_earning_private["observation_date"] = pd.to_datetime(average_hourly_earning_private["observation_date"], errors='ignore')
average_hourly_earning_private.columns
average_hourly_earning_private.rename(columns={"SMU08000000500000003":"averageHourlyEarningPrivate"}, inplace=True)

mortgage_rate_30y["observation_date"] = pd.to_datetime(mortgage_rate_30y["observation_date"], errors='ignore')
mortgage_rate_30y.columns
mortgage_rate_30y.rename(columns={'MORTGAGE30US':'30YearMortgageRate'}, inplace=True)
#Combining data together 

economic_df = pd.DataFrame(unemployment.merge(market_yield_on_10year_treasury))
economic_df = economic_df.merge(new_private_housing_unit_authorized)
economic_df = economic_df.merge(labor_force_participation_rate)
economic_df = economic_df.merge(consumer_sentiment_index)
economic_df = economic_df.merge(mortgage_rate_30y)
economic_df.rename(columns={"observation_date":"date"}, inplace=True)
economic_df.columns
cpi_df.columns= cpi_df.columns.str.lower()
economic_df

economic_df = economic_df.merge(cpi_df)
economic_df.columns
economic_df.head

homeValueIndex
heatIndex

#New Melted Data 






homeValueIndex_melted = melt_zillow(homeValueIndex,'homevalueindex')
homeValueIndex_melted = cleaning_regionNames(homeValueIndex_melted)
homeValueIndex_melted

observedRentIndex_melted = melt_zillow(observedRentIndex,'observedRentIndex')
observedRentIndex_melted = cleaning_regionNames(observedRentIndex_melted)
observedRentIndex_melted

new_construction_home_sold_melted = melt_zillow(new_construction_home_sold,"newConstructionHomeSold")
new_construction_home_sold_melted
new_construction_home_sold_melted = cleaning_regionNames(new_construction_home_sold_melted)
new_construction_home_sold_melted.isna().sum()
new_construction_home_sold.isna().sum()
print(new_construction_home_sold_melted.columns)

heatIndex_melted = melt_zillow(heatIndex,"HeatIndex")
heatIndex_melted = cleaning_regionNames(heatIndex_melted)
heatIndex_melted.columns
heatIndex_melted

forSale_inventory_melted = melt_zillow(forSale_inventory,"forSaleInventory")
forSale_inventory_melted = cleaning_regionNames(forSale_inventory_melted)
forSale_inventory_melted

df_zillow = homeValueIndex_melted.copy()
df_zillow
df_zillow = df_zillow.merge(observedRentIndex_melted, on=["date","RegionName"], how='outer')
df_zillow = df_zillow.merge(heatIndex_melted, on=["date","RegionName"], how='outer')
df_zillow = df_zillow.merge(forSale_inventory_melted, on=["date","RegionName"], how='outer')
df_zillow = df_zillow[~df_zillow['date'].isin(['2018-03-31', '2025-02-28'])]
df_zillow.columns
df_zillow
economic_df.columns

economic_df['date'] = pd.to_datetime(economic_df['date']) + pd.offsets.MonthEnd(0)
print(economic_df['date'].sort_values().unique())
final_df = df_zillow.merge(economic_df, on=["date"], how='left')
final_df.columns
final_df

big_df["RegionName"] = big_df["RegionName"].str.replace(", CO", "", regex=False).str.strip()
final_df["RegionName"] = final_df["RegionName"].str.replace(", CO", "", regex=False).str.strip()

big_df = big_df.merge(final_df,on=["date", "RegionName"], how='outer')
len(big_df.columns)

big_df 
big_df.to_csv("big_df_cleaned.csv", index=False)
big_df.to_excel("big_df_cleaned.xlsx", index=False)
df_matrix = big_df.dropna().copy()
df_matrix = df_matrix.drop(columns=["RegionName","date"])
corr_matrix = df_matrix.corr()
print(corr_matrix)
print(df_matrix.dtypes)

#plt.figure(figsize=(12,10))
#sns.heatmap(corr_matrix, annot=True, fmt="0.2f", cmap="coolwarm", linewidths=0.5)
#plt.title("Correlation Heatmap of Design Matrix")
#plt.tight_layout()
#plt.show()





final_df_regression = big_df.copy()
final_df_regression['date'] = pd.to_datetime(final_df_regression['date'])

# Add time features
final_df_regression['month_num'] = final_df_regression['date'].dt.month
final_df_regression['year'] = final_df_regression['date'].dt.year
final_df_regression['time_index'] = (final_df_regression['date'] - final_df_regression['date'].min()).dt.days

# Calculate inflation rate from CPI
final_df_regression["Inflation_Rate"] = final_df_regression["cpi"].pct_change() * 100
final_df_regression = final_df_regression.dropna()

# Store target variable before dropping


X = final_df_regression.drop(columns=["date",'new_homeOwnerIncomeNeeded_20Down','total_mortgage_payment_20Down','year','month_num','homevalueindex','observedRentIndex',"affordable_HomePrice_20_down",
                                      "new_homeOwnerIncomeNeeded_20Down","mortgage_payment_20Down","new_Renter_incomeNeeded","new_Renter_affordibility","30YearMortgageRate",
                                     "median_list_price","perccent_Of_homeSold_Below","new_PendingListing", "cpi", "total_transaction_value","median_sales_price",
                                     "new_homeOwnerAffordibility_20Down","perccent_Of_homeSold_Above",'share__of_listing_withPriceCut',"HeatIndex","new_construction_MedianSalesPrice_perSQF",
                                    "yearToSave_20Down","median_daySToClose_melted","RegionName"])
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
vif_Df = pd.DataFrame()
vif_Df["Feature"] = X.columns
vif_Df["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif_Df = vif_Df.sort_values(by='VIF', ascending =False)
print(vif_Df)

final_df_regression =final_df_regression.drop(columns=['new_homeOwnerIncomeNeeded_20Down','total_mortgage_payment_20Down','year','month_num','observedRentIndex',"affordable_HomePrice_20_down",
                                      "new_homeOwnerIncomeNeeded_20Down","mortgage_payment_20Down","new_Renter_incomeNeeded","new_Renter_affordibility","30YearMortgageRate",
                                     "median_list_price","perccent_Of_homeSold_Below","new_PendingListing", "cpi", "total_transaction_value","median_sales_price",
                                     "new_homeOwnerAffordibility_20Down","perccent_Of_homeSold_Above",'share__of_listing_withPriceCut',"HeatIndex","new_construction_MedianSalesPrice_perSQF",
                                    "yearToSave_20Down","median_daySToClose_melted","RegionName"])

final_df_regression.isna().sum()


# Split data for modeling
train_data = final_df_regression[final_df_regression['date'] <= '2022-12-31']
train_data_y = train_data["homevalueindex"]
train_data.columns
train_data_x = train_data.drop(columns=["homevalueindex",'date'])
train_data_x.columns

validation_data = final_df_regression[(final_df_regression['date'] >= '2023-01-01') & (final_df_regression['date'] <= '2023-12-31')]
validation_data_y = validation_data["homevalueindex"]
validation_data_x = validation_data.drop(columns=["homevalueindex",'date'])

test_data = final_df_regression[final_df_regression['date'] >= '2024-01-01']
test_data_y = test_data["homevalueindex"]
test_data_x = test_data.drop(columns=["homevalueindex",'date'])

print(train_data_x.isna().sum().sum())
print(validation_data_x.isna().sum().sum())
print(train_data_y.isna().sum())
print(validation_data_y.isna().sum())




#Training 
lr_model = LinearRegression()
lr_model.fit(train_data_x,train_data_y)

validation_prediction = lr_model.predict(validation_data_x)

validation_squared_errors = np.sqrt(mean_squared_error(validation_data_y, validation_prediction))
validation_mae = mean_absolute_error(validation_data_y, validation_prediction)
validation_r_2 = r2_score(validation_data_y,validation_prediction)

print(f"validation_squared_errors: {validation_squared_errors}")
print(" ")
print(f'Validation MAE: {validation_mae}')
print(" ")
print(f"validation_r_2 {validation_r_2}")


test_prediction = lr_model.predict(test_data_x)

test_squared_errors = np.sqrt(mean_squared_error(test_data_y, test_prediction))
test_mae = mean_absolute_error(test_data_y, test_prediction)
test_r_2 = r2_score(test_data_y,test_prediction)

print(f"test_squared_errors: {test_squared_errors}")
print(" ")
print(f'test_mae: {test_mae}')
print(" ")
print(f"test_r_2 {test_r_2}")




# Rent Model 

rent_df = big_df.copy()
rent_df.columns


rent_df["inflation_rate"] = rent_df["cpi"].pct_change() * 100
rent_df['date'] = pd.to_datetime(rent_df['date'])

# Add time features
rent_df['month_num'] = rent_df['date'].dt.month
rent_df['year'] = rent_df['date'].dt.year
rent_df['time_index'] = (rent_df['date'] - rent_df['date'].min()).dt.days

X_rent_features_VIF = rent_df.drop(columns=['RegionName','date','new_homeOwnerIncomeNeeded_20Down','total_mortgage_payment_20Down','year','month_num',
                                            'homevalueindex','mortgage_payment_20Down','new_homeOwnerAffordibility_20Down','30YearMortgageRate','cpi',
                                            'perccent_Of_homeSold_Above','perccent_Of_homeSold_Below',
                                            'total_transaction_value','median_sales_price','median_list_price','share__of_listing_withPriceCut','new_PendingListing','new_listings','median_daySToClose_melted',
                                            'median_daysToPending','new_construction_MedianSalesPrice_perSQF','median_SalesToList_Ration','observedRentIndex','sales_count_nowCast',
                                            'affordable_HomePrice_20_down','new_Renter_affordibility','new_contruction_Median_SalesPrice','ConsumerSentimentIndex','yearToSave_20Down'])
X_rent_features_VIF.columns
X_rent_features_VIF = X_rent_features_VIF.dropna()
rewnt_scaler = StandardScaler()
X_rent_scaled = rewnt_scaler.fit_transform(X_rent_features_VIF)

vif_rent_df = pd.DataFrame()
vif_rent_df["Feature"] = X_rent_features_VIF.columns
vif_rent_df["VIF"] = [variance_inflation_factor(X_rent_scaled, i) for i in range(X_rent_scaled.shape[1])]
vif_rent_df = vif_rent_df.sort_values(by='VIF', ascending =False)
print(vif_rent_df)

rent_regression_df = rent_df.drop(columns=['RegionName','new_homeOwnerIncomeNeeded_20Down','total_mortgage_payment_20Down','year','month_num',
                                            'homevalueindex','mortgage_payment_20Down','new_homeOwnerAffordibility_20Down','30YearMortgageRate','cpi',
                                            'perccent_Of_homeSold_Above','perccent_Of_homeSold_Below',
                                            'total_transaction_value','median_sales_price','median_list_price','share__of_listing_withPriceCut','new_PendingListing','new_listings','median_daySToClose_melted',
                                            'median_daysToPending','new_construction_MedianSalesPrice_perSQF','median_SalesToList_Ration','sales_count_nowCast','affordable_HomePrice_20_down',
                                            'new_Renter_affordibility','new_contruction_Median_SalesPrice','ConsumerSentimentIndex','yearToSave_20Down'])
rent_regression_df.columns
len(rent_regression_df.columns)

#lag_features = ['observedRentIndex', 'inflation_rate', 'time_index','unemploymentRate']

rent_regression_df['observedRentIndex_lag'] = rent_regression_df['observedRentIndex'].shift(1)
#rent_regression_df['time_lag1'] = rent_regression_df['time_index'].shift(1)
#rent_regression_df['time_lag2'] = rent_regression_df['time_index'].shift(2)
#rent_regression_df['inflation_rate_lag1'] = rent_regression_df['inflation_rate'].shift(1)
#rent_regressio_df[""]



#for feature in lag_features:
#    rent_regression_df[f"{feature}_lag1"] = rent_regression_df[feature].shift(1)

#for lag in [1, 2]:
    #for feature in lag_features:
    #    rent_regression_df[f"{feature}_lag{lag}"] = rent_regression_df[feature].shift(lag)



rent_regression_df = rent_regression_df.dropna()
rent_regression_df.columns

train_data_rent = rent_regression_df[rent_regression_df['date'] <= '2022-12-31']
train_data_rent =train_data_rent.dropna()
train_rent_y = train_data_rent['observedRentIndex']
train_data_rent.columns
train_rent_x = train_data_rent.drop(columns=['observedRentIndex','date'])
train_rent_x.columns

validation_data_rent = rent_regression_df [rent_regression_df['date'] >= '2023-01-01']
validation_data_rent.isna().sum()
validation_rent_y = validation_data_rent['observedRentIndex']
validation_rent_x = validation_data_rent.drop(columns=['observedRentIndex','date'])



rent_regression_df.isna().sum()
rent_regression_df = rent_regression_df.dropna()

rent_lr = LinearRegression()
rent_lr.fit(train_rent_x,train_rent_y)

valid_rent_predict = rent_lr.predict(validation_rent_x)

rent_validation_squared_errors = np.sqrt(mean_squared_error(validation_rent_y, valid_rent_predict))
rent_validation_mae = mean_absolute_error(validation_rent_y, valid_rent_predict)
rent_validation_r_2 = r2_score(validation_rent_y, valid_rent_predict)
def adjusted_r2(y_true, y_pred, X):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = X.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Compute Adjusted R²
rent_validation_adjusted_r2 = adjusted_r2(validation_rent_y, valid_rent_predict, validation_rent_x)

print(f"rent_validation_squared_errors: {rent_validation_squared_errors}")
print(" ")
print(f'rent_validation_mae: {rent_validation_mae}')
print(" ")
print(f"rent_validation_r_2: {rent_validation_r_2}")
print(" ")
print(f"Validation Adjusted R²: {rent_validation_adjusted_r2}")


validation_rent_x.columns

validation_rent_y.shape
validation_rent_x.shape
elastic_net_cv = ElasticNetCV(
    cv=5,
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1],  # range of l1_ratios to test
    n_alphas=100,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_rent_x)
X_valid_scaled = scaler.transform(validation_rent_x)

elastic_net_cv.fit(X_train_scaled, train_rent_y)
validation_pred_en = elastic_net_cv.predict(X_valid_scaled)

# Evaluate performance
validation_rmse_en = np.sqrt(mean_squared_error(validation_rent_y, validation_pred_en))
validation_mae_en = mean_absolute_error(validation_rent_y, validation_pred_en)
validation_r2_en = r2_score(validation_rent_y, validation_pred_en)

# Print results
print(f"ElasticNet Validation RMSE: {validation_rmse_en}")
print(f"ElasticNet Validation MAE: {validation_mae_en}")
print(f"ElasticNet Validation R²: {validation_r2_en}")
print(f"Best Alpha: {elastic_net_cv.alpha_}")
print(f"Best L1 Ratio: {elastic_net_cv.l1_ratio_}")




scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_rent_x)
X_val_scaled = scaler.transform(validation_rent_x)

# Ridge regression with hyperparameter tuning
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 100)}
ridge_cv = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
ridge_cv.fit(X_train_scaled, train_rent_y)

# Best Ridge model
best_ridge = ridge_cv.best_estimator_
val_ridge_preds = best_ridge.predict(X_val_scaled)

# Evaluation
ridge_rmse = np.sqrt(mean_squared_error(validation_rent_y, val_ridge_preds))
ridge_mae = mean_absolute_error(validation_rent_y, val_ridge_preds)
ridge_r2 = r2_score(validation_rent_y, val_ridge_preds)

# Adjusted R²
n = len(validation_rent_y)
p = validation_rent_x.shape[1]
ridge_adj_r2 = 1 - (1 - ridge_r2) * (n - 1) / (n - p - 1)

print(f"Ridge Validation RMSE: {ridge_rmse}")
print(f"Ridge Validation MAE: {ridge_mae}")
print(f"Ridge Validation R²: {ridge_r2}")
print(f"Ridge Adjusted R²: {ridge_adj_r2}")
print(f"Best Alpha: {ridge_cv.best_params_['alpha']}")

correlation_matrix = rent_regression_df.corr()

# Extract correlations with the target variable 'observedRentIndex'
rent_correlations = correlation_matrix['observedRentIndex'].drop('observedRentIndex')

# Sort correlations by absolute value (strongest to weakest)
rent_correlations = rent_correlations.reindex(rent_correlations.abs().sort_values(ascending=False).index)

# Display
print("Correlation of Features with observedRentIndex:\n")
print(rent_correlations)



X_rent_cv = rent_regression_df.drop(columns=["observedRentIndex", "date"])
y_rent_cv = rent_regression_df["observedRentIndex"]

# Scale features
scaler = StandardScaler()
X_rent_scaled_cv = scaler.fit_transform(X_rent_cv)

# ElasticNetCV with cross-validation
elastic_cv = ElasticNetCV(
    l1_ratio=np.linspace(0.1, 1, 10),
    alphas=np.logspace(-4, 1, 100),
    cv=5,
    max_iter=10000,
    random_state=42
)
elastic_cv.fit(X_rent_scaled_cv, y_rent_cv)

# R² score with 5-fold cross-validation
cv_scores_r2 = cross_val_score(
    elastic_cv, X_rent_scaled_cv, y_rent_cv, cv=5, scoring='r2'
)

# RMSE with custom scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)
cv_scores_rmse = cross_val_score(
    elastic_cv, X_rent_scaled_cv, y_rent_cv, cv=5, scoring=rmse_scorer
)

print("📊 ElasticNet Cross-Validation Results")
print(f"Mean CV R²: {cv_scores_r2.mean():.4f}")
print(f"Mean CV RMSE: {cv_scores_rmse.mean():.2f}")
print(f"Best Alpha: {elastic_cv.alpha_}")
print(f"Best L1 Ratio: {elastic_cv.l1_ratio_}")


X_sm = sm.add_constant(train_rent_x)

# Fit OLS model
ols_model = sm.OLS(train_rent_y, X_sm).fit()

# Extract diagnostics
fitted_vals = ols_model.fittedvalues
residuals = ols_model.resid
standardized_residuals = ols_model.get_influence().resid_studentized_internal
sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))

# Residuals vs Fitted
#plt.figure(figsize=(8, 5))
#sns.residplot(x=fitted_vals, y=train_rent_y, lowess=True, line_kws={'color': 'red'})
#plt.xlabel("Fitted values")
#plt.ylabel("Residuals")
#plt.title("Residuals vs Fitted")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

# Q-Q Plot
sm.qqplot(standardized_residuals, line='45')
plt.title("Normal Q-Q")
plt.tight_layout()
plt.show()

# Scale-Location Plot
#plt.figure(figsize=(8, 5))
#plt.scatter(fitted_vals, sqrt_std_resid)
#sns.regplot(x=fitted_vals, y=sqrt_std_resid, lowess=True, scatter=False, line_kws={'color': 'red'})
#plt.xlabel("Fitted values")
#plt.ylabel("√|Standardized Residuals|")
#plt.title("Scale-Location Plot")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.title("Histogram of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()



# 6. Durbin-Watson Test (Autocorrelation)
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Statistic (autocorrelation): {dw_stat:.3f}  ➜  ~2.0 is ideal")

# 7. Normality Tests
shapiro_stat, shapiro_p = shapiro(residuals)
jb_stat, jb_p = jarque_bera(residuals)
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}  ➜  {'OK' if shapiro_p > 0.05 else 'NOT normal'}")







rent_ts = rent_regression_df.dropna().copy()
rent_ts['date'] = pd.to_datetime(rent_ts['date'])
rent_ts.set_index('date', inplace=True)
rent_ts = rent_ts.asfreq('M')  # enforce monthly frequency

# Split into train/validation
train = rent_ts[rent_ts.index < '2023-01-01']
valid = rent_ts[rent_ts.index >= '2023-01-01']

# Fit SARIMA (can try plain ARIMA with order=(p,d,q) first)
model = SARIMAX(train['observedRentIndex'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
results = model.fit(disp=False)

# Predict
preds = results.predict(start=valid.index[0], end=valid.index[-1], typ='levels')

# Metrics
rmse = np.sqrt(mean_squared_error(valid['observedRentIndex'], preds))
mae = mean_absolute_error(valid['observedRentIndex'], preds)
r2 = r2_score(valid['observedRentIndex'], preds)
adjusted_r2 = 1 - (1 - r2) * (len(valid) - 1) / (len(valid) - 1 - 1)  # for 1 parameter

print(f"ARIMA RMSE: {rmse:.2f}")
print(f"ARIMA MAE: {mae:.2f}")
print(f"ARIMA R²: {r2:.4f}")
print(f"ARIMA Adjusted R²: {adjusted_r2:.4f}")

plt.figure(figsize=(12, 5))
plot_acf(rent_regression_df['observedRentIndex'], lags=30)
plt.show()

plt.figure(figsize=(12, 5))
plot_pacf(rent_regression_df['observedRentIndex'], lags=30)
plt.show()

result = adfuller(rent_regression_df['observedRentIndex'].dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


plt.plot(validation_rent_y.values, label='Actual')
plt.plot(valid_rent_predict, label='Linear/Ridge/ENet Prediction')
plt.plot(preds, label='ARIMA Prediction')  # for ARIMA model
plt.legend()
plt.title("Actual vs. Predicted Observed Rent Index")
plt.show()



rent_sarimax_df = rent_regression_df.dropna().copy()
rent_sarimax_df['date'] = pd.to_datetime(rent_sarimax_df['date'])
rent_sarimax_df.set_index('date', inplace=True)
rent_sarimax_df = rent_sarimax_df.asfreq('M')  # Ensure consistent frequency

rent_sarimax_df = rent_regression_df.dropna().copy()

# Split into train and validation
train_sarimax = rent_sarimax_df[rent_sarimax_df.index < '2023-01-01']
valid_sarimax = rent_sarimax_df[rent_sarimax_df.index >= '2023-01-01']



# Define endogenous (target) and exogenous
y_train = train_sarimax['observedRentIndex']
X_train = train_sarimax[['unemploymentRate', 'marketYeild10YTreasury', 'inflation_rate', 'forSaleInventory', 'LaborParticipationRate']]

y_valid = valid_sarimax['observedRentIndex']
X_valid = valid_sarimax[['unemploymentRate', 'marketYeild10YTreasury', 'inflation_rate', 'forSaleInventory', 'LaborParticipationRate']]

model_sarimax = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results_sarimax = model_sarimax.fit(disp=False)