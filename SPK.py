# Advanced NFL Player Analytics with PySpark SQL
# This notebook demonstrates advanced analytics on NFL player statistics using PySpark SQL

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("NFL Player Analytics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Create sample NFL player statistics dataset
nfl_data = [
    # (player_id, player_name, team, position, season, week, passing_yards, rushing_yards, receiving_yards, touchdowns, interceptions, fumbles, targets, receptions)
    (1, "Josh Allen", "BUF", "QB", 2023, 1, 314, 54, 0, 4, 0, 0, 0, 0),
    (1, "Josh Allen", "BUF", "QB", 2023, 2, 280, 12, 0, 2, 1, 0, 0, 0),
    (1, "Josh Allen", "BUF", "QB", 2023, 3, 325, 32, 0, 3, 0, 1, 0, 0),
    (2, "Stefon Diggs", "BUF", "WR", 2023, 1, 0, 0, 122, 1, 0, 0, 8, 7),
    (2, "Stefon Diggs", "BUF", "WR", 2023, 2, 0, 0, 98, 2, 0, 0, 10, 8),
    (2, "Stefon Diggs", "BUF", "WR", 2023, 3, 0, 0, 165, 1, 0, 0, 12, 11),
    (3, "Derrick Henry", "TEN", "RB", 2023, 1, 0, 128, 15, 2, 0, 0, 2, 1),
    (3, "Derrick Henry", "TEN", "RB", 2023, 2, 0, 95, 12, 0, 0, 1, 1, 1),
    (3, "Derrick Henry", "TEN", "RB", 2023, 3, 0, 142, 8, 1, 0, 0, 3, 2),
    (4, "Patrick Mahomes", "KC", "QB", 2023, 1, 360, 15, 0, 5, 1, 0, 0, 0),
    (4, "Patrick Mahomes", "KC", "QB", 2023, 2, 292, 8, 0, 2, 0, 0, 0, 0),
    (4, "Patrick Mahomes", "KC", "QB", 2023, 3, 331, 22, 0, 4, 1, 0, 0, 0),
    (5, "Tyreek Hill", "MIA", "WR", 2023, 1, 0, 5, 215, 2, 0, 0, 14, 11),
    (5, "Tyreek Hill", "MIA", "WR", 2023, 2, 0, 0, 181, 1, 0, 0, 13, 9),
    (5, "Tyreek Hill", "MIA", "WR", 2023, 3, 0, 8, 130, 0, 0, 0, 10, 7),
    (6, "Christian McCaffrey", "SF", "RB", 2023, 1, 0, 115, 45, 1, 0, 0, 4, 4),
    (6, "Christian McCaffrey", "SF", "RB", 2023, 2, 0, 89, 68, 2, 0, 0, 6, 5),
    (6, "Christian McCaffrey", "SF", "RB", 2023, 3, 0, 108, 32, 1, 0, 1, 3, 2),
]

# Define schema
schema = StructType([
    StructField("player_id", IntegerType(), True),
    StructField("player_name", StringType(), True),
    StructField("team", StringType(), True),
    StructField("position", StringType(), True),
    StructField("season", IntegerType(), True),
    StructField("week", IntegerType(), True),
    StructField("passing_yards", IntegerType(), True),
    StructField("rushing_yards", IntegerType(), True),
    StructField("receiving_yards", IntegerType(), True),
    StructField("touchdowns", IntegerType(), True),
    StructField("interceptions", IntegerType(), True),
    StructField("fumbles", IntegerType(), True),
    StructField("targets", IntegerType(), True),
    StructField("receptions", IntegerType(), True)
])

# Create DataFrame and register as temp view
df = spark.createDataFrame(nfl_data, schema)
df.createOrReplaceTempView("nfl_stats")

print("=== NFL Player Statistics Dataset ===")
df.show()

# 1. Advanced Analytics: Calculate Fantasy Points with Complex Scoring
print("\n=== 1. Fantasy Football Scoring Analysis ===")

fantasy_points_query = """
WITH fantasy_scoring AS (
    SELECT 
        player_name,
        team,
        position,
        week,
        -- Complex fantasy scoring algorithm
        CASE 
            WHEN position = 'QB' THEN 
                (passing_yards * 0.04) + (rushing_yards * 0.1) + (touchdowns * 6) - (interceptions * 2) - (fumbles * 2)
            WHEN position = 'RB' THEN 
                (rushing_yards * 0.1) + (receiving_yards * 0.1) + (touchdowns * 6) - (fumbles * 2) + (receptions * 0.5)
            WHEN position = 'WR' THEN 
                (receiving_yards * 0.1) + (rushing_yards * 0.1) + (touchdowns * 6) - (fumbles * 2) + (receptions * 1)
            ELSE 0
        END as fantasy_points,
        passing_yards,
        rushing_yards,
        receiving_yards,
        touchdowns,
        receptions
    FROM nfl_stats
),
weekly_rankings AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY position, week ORDER BY fantasy_points DESC) as position_rank,
        PERCENT_RANK() OVER (PARTITION BY position, week ORDER BY fantasy_points DESC) as percentile_rank
    FROM fantasy_scoring
)
SELECT 
    player_name,
    team,
    position,
    week,
    ROUND(fantasy_points, 2) as fantasy_points,
    position_rank,
    ROUND(percentile_rank * 100, 1) as percentile
FROM weekly_rankings
WHERE position_rank <= 5
ORDER BY position, week, position_rank
"""

spark.sql(fantasy_points_query).show(50)

# 2. Advanced Window Functions: Rolling Averages and Trend Analysis
print("\n=== 2. Performance Trends and Rolling Statistics ===")

trend_analysis_query = """
WITH player_stats AS (
    SELECT 
        player_name,
        position,
        week,
        passing_yards + rushing_yards + receiving_yards as total_yards,
        touchdowns,
        CASE WHEN targets > 0 THEN ROUND(receptions * 100.0 / targets, 1) ELSE 0 END as catch_rate
    FROM nfl_stats
),
rolling_stats AS (
    SELECT *,
        -- 3-week rolling average
        AVG(total_yards) OVER (
            PARTITION BY player_name 
            ORDER BY week 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as rolling_avg_yards,
        -- Cumulative statistics
        SUM(total_yards) OVER (
            PARTITION BY player_name 
            ORDER BY week
        ) as cumulative_yards,
        SUM(touchdowns) OVER (
            PARTITION BY player_name 
            ORDER BY week
        ) as cumulative_tds,
        -- Week-over-week change
        total_yards - LAG(total_yards) OVER (
            PARTITION BY player_name 
            ORDER BY week
        ) as yards_change,
        -- Performance consistency (coefficient of variation)
        STDDEV(total_yards) OVER (
            PARTITION BY player_name 
            ORDER BY week 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) / NULLIF(AVG(total_yards) OVER (
            PARTITION BY player_name 
            ORDER BY week 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 0) as consistency_score
    FROM player_stats
)
SELECT 
    player_name,
    position,
    week,
    total_yards,
    ROUND(rolling_avg_yards, 1) as rolling_avg_yards,
    cumulative_yards,
    cumulative_tds,
    COALESCE(yards_change, 0) as yards_change,
    ROUND(COALESCE(consistency_score, 0), 3) as consistency_score
FROM rolling_stats
ORDER BY player_name, week
"""

spark.sql(trend_analysis_query).show(50)

# 3. Advanced Team Analytics: Offensive Efficiency and Player Contributions
print("\n=== 3. Team Offensive Analysis ===")

team_analysis_query = """
WITH team_totals AS (
    SELECT 
        team,
        SUM(passing_yards + rushing_yards + receiving_yards) as total_team_yards,
        SUM(touchdowns) as total_team_tds,
        COUNT(DISTINCT player_id) as players_used,
        AVG(CASE WHEN targets > 0 THEN receptions * 100.0 / targets END) as avg_catch_rate
    FROM nfl_stats
    GROUP BY team
),
player_contributions AS (
    SELECT 
        s.team,
        s.player_name,
        s.position,
        SUM(s.passing_yards + s.rushing_yards + s.receiving_yards) as player_total_yards,
        SUM(s.touchdowns) as player_touchdowns,
        t.total_team_yards,
        t.total_team_tds
    FROM nfl_stats s
    JOIN team_totals t ON s.team = t.team
    GROUP BY s.team, s.player_name, s.position, t.total_team_yards, t.total_team_tds
)
SELECT 
    team,
    player_name,
    position,
    player_total_yards,
    player_touchdowns,
    ROUND(player_total_yards * 100.0 / total_team_yards, 1) as yards_share_pct,
    ROUND(player_touchdowns * 100.0 / NULLIF(total_team_tds, 0), 1) as td_share_pct,
    -- Efficiency metrics
    ROUND(player_total_yards / 3.0, 1) as yards_per_game,
    ROUND(player_touchdowns / 3.0, 2) as tds_per_game
FROM player_contributions
WHERE player_total_yards > 0
ORDER BY team, yards_share_pct DESC
"""

spark.sql(team_analysis_query).show()

# 4. Advanced Position Analytics: Multi-dimensional Performance Comparison
print("\n=== 4. Position-Specific Advanced Metrics ===")

position_analytics_query = """
WITH position_metrics AS (
    SELECT 
        position,
        player_name,
        -- QB-specific metrics
        CASE WHEN position = 'QB' THEN
            SUM(passing_yards) / NULLIF(SUM(CASE WHEN passing_yards > 0 THEN 1 END), 0)
        END as avg_passing_yards_per_game,
        CASE WHEN position = 'QB' THEN
            SUM(touchdowns) * 1.0 / NULLIF(SUM(interceptions + CASE WHEN interceptions = 0 THEN 1 ELSE 0 END), 0)
        END as td_to_int_ratio,
        
        -- RB-specific metrics  
        CASE WHEN position = 'RB' THEN
            (SUM(rushing_yards) + SUM(receiving_yards)) / 3.0
        END as rb_total_yards_per_game,
        CASE WHEN position = 'RB' THEN
            SUM(receiving_yards) * 100.0 / NULLIF(SUM(rushing_yards + receiving_yards), 0)
        END as receiving_yards_pct,
        
        -- WR-specific metrics
        CASE WHEN position = 'WR' THEN
            SUM(receptions) * 1.0 / NULLIF(SUM(targets), 0)
        END as catch_rate,
        CASE WHEN position = 'WR' THEN
            SUM(receiving_yards) * 1.0 / NULLIF(SUM(receptions), 0)
        END as yards_per_reception,
        
        -- Universal metrics
        SUM(touchdowns) as total_tds,
        (SUM(passing_yards) + SUM(rushing_yards) + SUM(receiving_yards)) / 3.0 as total_yards_per_game
    FROM nfl_stats
    GROUP BY position, player_name
),
position_rankings AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY position ORDER BY total_yards_per_game DESC) as yards_rank,
        ROW_NUMBER() OVER (PARTITION BY position ORDER BY total_tds DESC) as td_rank,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY total_yards_per_game DESC) as yards_percentile
    FROM position_metrics
)
SELECT 
    position,
    player_name,
    ROUND(total_yards_per_game, 1) as yards_per_game,
    total_tds,
    yards_rank,
    td_rank,
    ROUND(yards_percentile * 100, 1) as yards_percentile,
    -- Position-specific metrics
    ROUND(COALESCE(avg_passing_yards_per_game, 0), 1) as qb_pass_ypg,
    ROUND(COALESCE(td_to_int_ratio, 0), 2) as qb_td_int_ratio,
    ROUND(COALESCE(rb_total_yards_per_game, 0), 1) as rb_total_ypg,
    ROUND(COALESCE(receiving_yards_pct, 0), 1) as rb_receiving_pct,
    ROUND(COALESCE(catch_rate * 100, 0), 1) as wr_catch_rate_pct,
    ROUND(COALESCE(yards_per_reception, 0), 1) as wr_ypr
FROM position_rankings
ORDER BY position, yards_rank
"""

spark.sql(position_analytics_query).show()

# 5. Complex Aggregations: Multi-level Grouping and Statistical Analysis
print("\n=== 5. Statistical Performance Distribution Analysis ===")

statistical_analysis_query = """
WITH performance_stats AS (
    SELECT 
        position,
        team,
        player_name,
        (passing_yards + rushing_yards + receiving_yards) as total_yards,
        touchdowns,
        CASE 
            WHEN (passing_yards + rushing_yards + receiving_yards) >= 150 THEN 'High'
            WHEN (passing_yards + rushing_yards + receiving_yards) >= 75 THEN 'Medium' 
            ELSE 'Low'
        END as performance_tier
    FROM nfl_stats
),
statistical_summary AS (
    SELECT 
        position,
        performance_tier,
        COUNT(*) as games_count,
        ROUND(AVG(total_yards), 1) as avg_yards,
        ROUND(STDDEV(total_yards), 1) as stddev_yards,
        MIN(total_yards) as min_yards,
        MAX(total_yards) as max_yards,
        ROUND(PERCENTILE_APPROX(total_yards, 0.5), 1) as median_yards,
        ROUND(PERCENTILE_APPROX(total_yards, 0.75), 1) as q3_yards,
        ROUND(PERCENTILE_APPROX(total_yards, 0.25), 1) as q1_yards,
        ROUND(AVG(touchdowns), 2) as avg_touchdowns
    FROM performance_stats
    GROUP BY position, performance_tier
),
position_summary AS (
    SELECT 
        position,
        COUNT(*) as total_games,
        COUNT(DISTINCT CASE WHEN performance_tier = 'High' THEN 1 END) as high_perf_games,
        ROUND(COUNT(DISTINCT CASE WHEN performance_tier = 'High' THEN 1 END) * 100.0 / COUNT(*), 1) as high_perf_pct
    FROM performance_stats
    GROUP BY position
)
SELECT 
    s.position,
    s.performance_tier,
    s.games_count,
    s.avg_yards,
    s.stddev_yards,
    s.min_yards,
    s.median_yards,
    s.max_yards,
    s.q1_yards,
    s.q3_yards,
    s.avg_touchdowns,
    p.high_perf_pct as position_high_perf_pct
FROM statistical_summary s
JOIN position_summary p ON s.position = p.position
ORDER BY s.position, 
         CASE s.performance_tier 
             WHEN 'High' THEN 1 
             WHEN 'Medium' THEN 2 
             ELSE 3 
         END
"""

spark.sql(statistical_analysis_query).show()

# 6. Advanced Correlation Analysis
print("\n=== 6. Performance Correlation Analysis ===")

correlation_query = """
WITH weekly_team_stats AS (
    SELECT 
        team,
        week,
        SUM(CASE WHEN position = 'QB' THEN passing_yards END) as team_passing_yards,
        SUM(CASE WHEN position = 'RB' THEN rushing_yards END) as team_rushing_yards,
        SUM(CASE WHEN position IN ('WR', 'RB') THEN receiving_yards END) as team_receiving_yards,
        SUM(touchdowns) as team_touchdowns,
        COUNT(DISTINCT player_id) as players_active
    FROM nfl_stats
    GROUP BY team, week
),
team_balance_metrics AS (
    SELECT *,
        team_passing_yards + team_rushing_yards + team_receiving_yards as total_offense,
        CASE 
            WHEN team_passing_yards > team_rushing_yards + team_receiving_yards THEN 'Pass Heavy'
            WHEN team_rushing_yards > team_passing_yards * 0.6 THEN 'Run Heavy'
            ELSE 'Balanced'
        END as offensive_style,
        team_rushing_yards * 1.0 / NULLIF(team_passing_yards, 0) as run_pass_ratio
    FROM weekly_team_stats
)
SELECT 
    team,
    offensive_style,
    COUNT(*) as games,
    ROUND(AVG(total_offense), 1) as avg_total_offense,
    ROUND(AVG(team_touchdowns), 1) as avg_touchdowns,
    ROUND(AVG(run_pass_ratio), 2) as avg_run_pass_ratio,
    ROUND(AVG(players_active), 1) as avg_players_used,
    -- Efficiency metrics
    ROUND(AVG(total_offense) / AVG(players_active), 1) as yards_per_player,
    ROUND(AVG(team_touchdowns) * 1.0 / AVG(total_offense) * 100, 2) as td_efficiency_pct
FROM team_balance_metrics
GROUP BY team, offensive_style
ORDER BY avg_total_offense DESC
"""

spark.sql(correlation_query).show()

print("\n=== Analysis Complete! ===")
print("This notebook demonstrated:")
print("• Complex fantasy scoring calculations")
print("• Window functions for rolling averages and rankings") 
print("• Multi-dimensional team analysis")
print("• Position-specific advanced metrics")
print("• Statistical distribution analysis")
print("• Performance correlation patterns")

# Clean up
spark.stop()
