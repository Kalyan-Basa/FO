from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, DateType
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import requests
from datetime import datetime, timedelta
import os
from io import BytesIO
from zipfile import ZipFile
import pandas as pd

class Bhavcopy:
    def __init__(self):
        self.base_url = "https://nsearchives.nseindia.com/content/fo"
        self.output_folder = "/Volumes/trading_data/default/series-vwap/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()

    def get_trading_dates(self, days_back=10):
        dates = []
        current_date = datetime.now()
        while len(dates) < days_back:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date -= timedelta(days=1)
        return dates

    def fetch_bhavcopy(self, date):
        try:
            date_str = date.strftime('%Y%m%d')
            filename = f"Bhavcopy_{date_str}.parquet"
            parquet_path = self.output_folder + filename

            if os.path.exists(parquet_path):
                print(f"File {filename} already exists. Skipping download.")
                return spark.read.parquet(parquet_path)

            zip_file_name = f'BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip'
            url = f"{self.base_url}/{zip_file_name}"
            print(f"Fetching data for {url} {date.strftime('%Y-%m-%d')}...")

            self.session.get('https://www.nseindia.com', headers=self.headers, timeout=30)
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            with ZipFile(BytesIO(response.content)) as zip_file:
                csv_filename = zip_file.namelist()[0]
                with zip_file.open(csv_filename) as csv_file:
                    df_pd = pd.read_csv(csv_file)

            df_pd = df_pd[df_pd['FinInstrmTp'] == 'STF']

            print(f"{filename} - Filtered to {len(df_pd)} records for Futures")

            # Convert pandas DataFrame to Spark DataFrame and cast columns
            spark_df = spark.createDataFrame(df_pd)

            # Ensure columns are cast to correct data types
            futures_df = spark_df.withColumn("TIMESTAMP", F.to_date(F.col("BizDt"))) \
                                 .withColumn("OPEN", F.col("OpnPric").cast("double")) \
                                 .withColumn("CLOSE", F.col("ClsPric").cast("double")) \
                                 .withColumn("HIGH", F.col("HghPric").cast("double")) \
                                 .withColumn("LOW", F.col("LwPric").cast("double")) \
                                 .withColumn("CONTRACTS", F.col("TtlTradgVol").cast("long")) \
                                 .withColumn("SYMBOL", F.col("TckrSymb")) \
                                 .withColumn("EXPIRY_DT", F.to_date(F.col("XpryDt"))) \
                                 .withColumn("OPEN_INT", F.col("OpnIntrst").cast("long"))

            results = futures_df.select("TIMESTAMP", "OPEN", "CLOSE", "HIGH", "LOW", "CONTRACTS", "SYMBOL", "EXPIRY_DT", "OPEN_INT")

            # 3. Calculate Typical Price for VWAP Calculation
            # Typical Price = (High + Low + Close) / 3
            results = results.withColumn("TYPICAL_PRICE", (F.col("HIGH") + F.col("LOW") + F.col("CLOSE")) / 3)
                        
            # Calculate Price * Volume (using CONTRACTS as proxy for volume)
            results = results.withColumn("PRICE_X_VOL", F.col("TYPICAL_PRICE") * F.col("CONTRACTS"))

            # Save as parquet
            results.write.mode("overwrite").parquet(parquet_path)

            return results

        except Exception as e:
            print(f"Error fetching data for {date.strftime('%Y-%m-%d')}: {str(e)}")
            return None
        
    def AnalyseData(self, date):
        # 1. Initialize Spark Session
        spark = SparkSession.builder.appName("BhavCopy_Series_VWAP_Analysis").getOrCreate()
        
        # Load your Bhav Copy data (Update the path to your actual CSV files)
        # Assuming date format in CSV is yyyy-MM-dd
        results = spark.read.parquet("/Volumes/trading_data/default/series-vwap/*.parquet", header=True, inferSchema=True)

        # 4. Define Windows for Analysis
        # Window for daily day-over-day changes (Partitioned by Symbol and Expiry)
        daily_window = Window.partitionBy("SYMBOL", "EXPIRY_DT").orderBy("TIMESTAMP")

        # Window for Cumulative Series VWAP (Expanding window from the start of the series)
        series_window = Window.partitionBy("SYMBOL", "EXPIRY_DT") \
                            .orderBy("TIMESTAMP") \
                            .rowsBetween(Window.unboundedPreceding, Window.currentRow)

        # 5. Apply Calculations
        analysis_df = results.withColumn("PREV_CLOSE", F.lag("CLOSE").over(daily_window)) \
                                .withColumn("PREV_OI", F.lag("OPEN_INT").over(daily_window)) \
                                .withColumn("CUM_VOL", F.sum("CONTRACTS").over(series_window)) \
                                .withColumn("CUM_PRICE_X_VOL", F.sum("PRICE_X_VOL").over(series_window))

        # Calculate Series VWAP
        analysis_df = analysis_df.withColumn(
            "SERIES_VWAP", 
            F.try_divide(F.col("CUM_PRICE_X_VOL"), F.col("CUM_VOL"))
        )

        # 6. Identify Trend based on Price and OI relationship
        # Long Buildup: Price UP, OI UP
        # Short Buildup: Price DOWN, OI UP
        # Short Covering: Price UP, OI DOWN
        # Long Unwinding: Price DOWN, OI DOWN
        analysis_df = analysis_df.withColumn(
            "TREND",
            F.when((F.col("CLOSE") > F.col("PREV_CLOSE")) & (F.col("OPEN_INT") > F.col("PREV_OI")), "Long Buildup")
            .when((F.col("CLOSE") < F.col("PREV_CLOSE")) & (F.col("OPEN_INT") > F.col("PREV_OI")), "Short Buildup")
            .when((F.col("CLOSE") > F.col("PREV_CLOSE")) & (F.col("OPEN_INT") < F.col("PREV_OI")), "Short Covering")
            .when((F.col("CLOSE") < F.col("PREV_CLOSE")) & (F.col("OPEN_INT") < F.col("PREV_OI")), "Long Unwinding")
            .otherwise("Neutral/No Change")
        )

        # 7. Generate Actionable Signals (Pankaj's Setup)
        # Buy Signal: Price crosses above Series VWAP + Long Buildup
        # Sell Signal: Price crosses below Series VWAP + Short Buildup
        analysis_df = analysis_df.withColumn(
            "SIGNAL",
            F.when((F.col("CLOSE") > F.col("SERIES_VWAP")) & (F.col("TREND") == "Long Buildup"), "BULLISH - Above VWAP")
            .when((F.col("CLOSE") < F.col("SERIES_VWAP")) & (F.col("TREND") == "Short Buildup"), "BEARISH - Below VWAP")
            .otherwise("WAIT")
        )

        # Select final columns for review
        final_output = analysis_df.select(
            "TIMESTAMP", "SYMBOL", "EXPIRY_DT", "CLOSE", "PREV_CLOSE", 
            "OPEN_INT", "PREV_OI", "SERIES_VWAP", "TREND", "SIGNAL"
        ).filter(F.col("TIMESTAMP") == date.strftime("%Y-%m-%d")).orderBy("TIMESTAMP", F.desc("OPEN_INT"))

        final_output.write.mode("overwrite").parquet("/Volumes/trading_data/default/results_series_vwap/series_vwap_" + date.strftime("%Y%m%d") + ".parquet")        
       
