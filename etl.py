import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (udf, col, monotonically_increasing_id,
                                   year, month, dayofmonth, hour,
                                   weekofyear, dayofweek)
from pyspark.sql.types import (StructType, StructField, IntegerType,
                               StringType, FloatType, LongType, DoubleType)


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'


def create_spark_session():
    """Creates and returns SparkSession instance"""
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.3") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Reads in song data, selects columns for songs_table
       and artists_table, writes songs_table and artists_table
       to S3 in parquet format.

       Parameters
       ----------
       spark: SparkSession instance
       input_data: str
           root location of input data in S3
       output_data: str
           root location of output data in S3
    """
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    # song_data = input_data + 'song_data/A/B/C/*.json'

    # define song schema
    song_schema = StructType([
        StructField("num_songs", IntegerType(), True),
        StructField("artist_id", StringType(), True),
        StructField("artist_latitude", FloatType(), True),
        StructField("artist_longitude", FloatType(), True),
        StructField("artist_location", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("song_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("duration", FloatType(), True),
        StructField("year", IntegerType(), True),
    ])
    # read song data file
    df = spark.read.json(song_data, schema=song_schema)

    # extract columns to create songs table
    songs_cols = ['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = df.select(*songs_cols) \
        .filter(col("artist_id").isNotNull()) \
        .filter(col("artist_name").isNotNull()) \
        .filter(col("song_id").isNotNull()) \
        .filter(col("title").isNotNull()) \
        .filter(col("duration").isNotNull()) \
        .dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_path = output_data + 'songs_table/songs_table.parquet'
    songs_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'artist_id') \
        .parquet(songs_path)

    # extract columns to create artists table
    artists_fields = [col('artist_id'),
                      col('artist_name').alias('name'),
                      col('artist_location').alias('location'),
                      col('artist_latitude').alias('latitude'),
                      col('artist_longitude').alias('longitude')]
    artists_table = df.select(*artists_fields) \
        .filter(col("artist_id").isNotNull()) \
        .filter(col("artist_name").isNotNull()) \
        .dropDuplicates()

    # write artists table to parquet files
    artists_path = output_data + 'artists_table/artists_table.parquet'
    artists_table.write \
        .mode('overwrite') \
        .parquet(artists_path)


def process_log_data(spark, input_data, output_data):
    """Reads in log data and songs_table, creates users_table,
       time_table, and songplays_table. Writes all three to S3
       in parquet format.

       Parameters
       ----------
       spark: SparkSession instance
       input_data: str
           root location of input data in S3
       output_data: str
           root location of output data in S3
    """

    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    log_schema = StructType([
        StructField("artist", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("itemInSession", LongType(), True),
        StructField("lastName", StringType(), True),
        StructField("length", DoubleType(), True),
        StructField("level", StringType(), True),
        StructField("location", StringType(), True),
        StructField("method", StringType(), True),
        StructField("page", StringType(), True),
        StructField("registration", DoubleType(), True),
        StructField("sessionId", LongType(), True),
        StructField("song", StringType(), True),
        StructField("status", LongType(), True),
        StructField("ts", LongType(), True),
        StructField("userAgent", StringType(), True),
        StructField("userId", StringType(), True),
    ])

    # read log data file
    log_cols = ['userId', 'firstName', 'lastName', 'gender', 'ts',
                'level', 'sessionId', 'location', 'userAgent', 'song']
    df = spark.read \
        .json(log_data, schema=log_schema) \
        .select(*log_cols) \
        .filter("page = 'NextSong'")

    # extract columns for users table
    users_columns = [col('userId').alias('user_id'),
                     col('firstName').alias('first_name'),
                     col('lastName').alias('last_name'),
                     col('gender'),
                     col('level')]
    users_table = df.select(*users_columns) \
        .filter(col('user_id').isNotNull()) \
        .filter(col('first_name').isNotNull()) \
        .filter(col('last_name').isNotNull()) \
        .filter(col('level').isNotNull()) \
        .dropDuplicates()

    # write users table to parquet files
    users_path = output_data + 'users_table/users_table.parquet'
    users_table.write \
        .mode('overwrite') \
        .parquet(users_path)

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: round(x/1000))
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x)
                       .strftime("%Y-%m-%d %H:%M:%S"))
    df = df.withColumn('datetime', get_datetime(df.timestamp))

    # extract columns to create time table
    # start_time, hour, day, week, month, year, weekday
    time_cols = [col('ts').alias('start_times'),
                 hour('datetime').alias('hour'),
                 dayofmonth('datetime').alias('day'),
                 weekofyear('datetime').alias('week'),
                 month('datetime').alias('month'),
                 year('datetime').alias('year'),
                 dayofweek('datetime').alias('weekday')]
    time_table = df.select(*time_cols) \
        .filter(col('ts').isNotNull()) \
        .dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_path = output_data + 'time_table/time_table.parquet'
    time_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(time_path)

    # read in song data to use for songplays table
    # 'song_id', 'title', 'artist_id', 'year', 'duration'
    songs_path = output_data + 'songs_table/songs_table.parquet'
    song_df = spark.read.parquet(songs_path)

    # get columns from joined song and log datasets to create songplays table
    # add year and month for partitioning
    songplays_table = df.alias('l').join(song_df.alias('s'),
                                         on=col('l.song') == col('s.title')) \
        .select([monotonically_increasing_id().alias('songplay_id'),
                 col('l.ts').alias('start_time'),
                 col('l.userId').alias('user_id'),
                 col('l.level'),
                 col('s.song_id'),
                 col('s.artist_id'),
                 col('l.sessionId').alias('session_id'),
                 col('l.location'),
                 col('l.userAgent').alias('user_agent'),
                 year(col('l.datetime')).alias('year'),
                 month(col('l.datetime')).alias('month')])

    # write songplays table to parquet files partitioned by year and month
    songplays_path = output_data + 'songplays_table/songplays_table.parquet'
    songplays_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(songplays_path)


def main():
    """Creates SparkSessionInstance, defines input_data
       and output_data, calls process_song_data() and
       process_log_data().
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://udacity-sparkify-datalake/analytics/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
