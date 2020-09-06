import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (udf, col, monotonically_increasing_id,
                                   year, month, dayofmonth, hour,
                                   weekofyear, dayofweek)


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2") \
        .getOrCreate()
    # spark.conf.set('mapreduce.fileoutputcommitter.algorithm.version', '2')
    return spark


def process_song_data(spark, input_data, output_data):
    """INSERT DOCSTRING HERE"""
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*'

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_cols = ['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = df.select(*songs_cols).dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_path = output_data + 'songs_table/songs_table.parquet'
    songs_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'artist_id') \
        .parquet(songs_path)


def process_artist_data(spark, input_data, output_data):
    """INSERT DOCSTRING HERE"""
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*'

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create artists table
    artists_fields = ['artist_id', 'artist_name as name',
                      'artist_location as location',
                      'artist_latitude as latitude',
                      'artist_longitude as longitude']
    artists_table = df.selectExpr(*artists_fields).dropDuplicates()

    # write artists table to parquet files
    artists_path = output_data + 'artists_table/artists_table.parquet'
    artists_table.write \
        .mode('overwrite') \
        .parquet(artists_path)


def process_songplay_data(spark, input_data, output_data):

    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*'

    # read log data file
    # drop schema
    df = spark.read.json(log_data).filter("page = 'NextSong'")

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
        .partitionBy('year', 'month') \
        .parquet(songplays_path, mode='overwrite')


def process_user_data(spark, input_data, output_data):

    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*'

    # read log data file
    # drop schema
    df = spark.read.json(log_data).filter("page = 'NextSong'")

    # extract columns for users table
    users_columns = ['userId as user_id', 'firstName as first_name',
                     'lastName as last_name', 'gender', 'level']
    users_table = df.selectExpr(*users_columns).dropDuplicates()

    # write users table to parquet files
    users_path = output_data + 'users_table/users_table.parquet'
    users_table.write \
        .parquet(users_path, mode='overwrite')


def process_time_data(spark, input_data, output_data):

    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*'

    # read log data file
    # drop schema
    df = spark.read.json(log_data).filter("page = 'NextSong'")

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
    time_table = df.select(*time_cols).dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_path = output_data + 'time_table/time_table.parquet'
    time_table.write \
        .partitionBy('year', 'month') \
        .parquet(time_path, mode='overwrite')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-sparkify-datalake/"

    process_song_data(spark, input_data, output_data)
    process_artist_data(spark, input_data, output_data)
    process_songplay_data(spark, input_data, output_data)
    process_user_data(spark, input_data, output_data)
    process_time_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
