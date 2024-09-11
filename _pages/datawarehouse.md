---
permalink: /portfolio/datawarehouse/
title: "A Real Time Streaming Data Warehouse with Reporting"
# excerpt: ""
header:
  overlay_image: /assets/images/lukasz-rawa-IVJiTrJfNjA-unsplash.jpg  
  caption: "Photo credit: [Lukasz Rawa](https://unsplash.com/@agathe_26) on [Unsplash](https://unsplash.com)"
last_modified_at: 2024-09-05T11:59:26-04:00
author_profile: true
layout: single
intro: 
  - excerpt: 'A journey through the challenges and triumphs when building a real-time streaming data warehouse.'
toc: true
toc_label: "Table of Contents"
toc_sticky: true
---
{% include feature_row id="intro" type="center" %}

# Background
Our company only had a transactional data warehouse, but they needed a data warehouse that collected and contained historical data, so they could create advanced analytical reports and develop machine learning models. To address this, me and my team developed a data warehouse that captured transactional data in real-time, stored it, and were made availabile immediately for reporting.

# Process
To build the data warehouse, I guided a portion of the team in building data pipelines from our transactional (OLTP) data warehouse in MariaDB to Confluent Kafka. Additionally, I guided them in the pipelines from Kafka to S3.

Simultaneously, I guided another portion of the team in building the infrastructure for Kafka.

Finally, at the same time, I guided the last portion of the team in building the pipeline from S3 to the data warehouse. Additionally, I guided them in building the analytical data warehouse (OLAP) in Databricks.

![image-center](/assets/images/data pipeline.png){: .align-center}

During this process, there were several challenges that occured, where I had to convene with leadership, stakeholders, and the team to come to be the best solution possible given the budget, resources, and time constraints. 

# Challenge 1
## Challenge
There were hundreds of clients, each having their own database.

This resulted in a situation where each client could potentially have their own unique data warehouse with their own data model and schema. As a result, this would limit how the new data warehouse would scale as more clients were ingested. 

## Solution
Move the company to a standardized data model.

This required many discussions and buy-in from leadership as well as the teams in infrastructure, dev ops, product, and account. This entire process required several months of discussions to get everyone on board.


[![image-right](/assets/images/sample alert.jpg){: .align-right}](https://docs.databricks.com/en/sql/user/alerts/index.html) In the meantime, inside of Databricks, I built an [automated alerting system](https://docs.databricks.com/en/sql/user/alerts/index.html) using python and SQL that assessed each client's data tables and schemas in production and staging to ensure of any changes. If there were any discrepencies from staging to production, our team would be alerted, and we would be able to account for those changes in the pipeline and the OLAP data warehouse.

## Result
The alerting system resulted in zero downtime for the OLAP data warehouse, and the unified model ensured consistency and scalability.

# Challenge 2
## Challenge
Ingesting new clients required the dedication of 2 - 3 resources over 2 - 4 weeks.

This hampered scalability and the feasbility of ingesting many clients at once.

## Solution
Guide the team in designing an infrastructure as code (IaC) solution inside of Databricks that automated new client configurations and deployment for the data warehouse, jobs, and reports.

![image-center](/assets/images/IaC.png){: .align-center}

Here's a stripped down sample of the code I guided the team to build after several discussions:
```python
# config file
meta_tbl_df = spark.sql(
    """select * from {}.{}.{}""".format(META_CATALOG, META_SCHEMA, meta_table)
).cache()
table_catalog = meta_tbl_df.select("table_catalog").collect()[0][0]
tab_schema = meta_tbl_df.select("table_schema").collect()[0][0]
tables_from_meta = set(
    meta_tbl_df.select(collect_set("table_name")).collect()[0][0]
)

# client dependent tables to create
available_tables = set(
    spark.sql(
        """select collect_set(upper(table_name)) from {}.information_schema.tables where table_schema = '{}'""".format(
            table_catalog, tab_schema
        )
    ).collect()[0][0]
)
create_table_list = list(tables_from_meta - available_tables)

# configs for each of the client dependent tables
for tab in create_table_list:
    tab_cols = (
        meta_tbl_df.filter(meta_tbl_df["TABLE_NAME"] == tab.upper())
        .selectExpr(
            "collect_list(concat(COLUMN_NAME, ' ', DATA_TYPE, ' ', (case when IS_NULLABLE = 'NO' then 'NOT NULL' else '' end)))"
        )
        .collect()[0][0]
    )
    CONFIG_OPTION_1 = (
        meta_tbl_df.select(upper("CONFIG_OPTION_1"))
        .filter(meta_tbl_df["TABLE_NAME"] == tab.upper())
        .collect()[0][0]
    )
    CONFIG_OPTION_2 = (
        meta_tbl_df.select("CONFIG_OPTION_2")
        .filter(meta_tbl_df["TABLE_NAME"] == tab.upper())
        .collect()[0][0]
    )

    # create client dependent tables based on configs
    create_table = (
        "CREATE TABLE IF NOT EXISTS "
        + qualified_name
        + " (\n"
        + ",\n".join(tab_cols)
        + ")USING DELTA TBLPROPERTIES(DELTA.ENABLECHANGEDATAFEED = {}, delta.columnMapping.mode = '{}', delta.minReaderVersion = '{}', delta.minWriterVersion = '{}');".format(
            CONFIG_OPTION_1,
            CONFIG_OPTION_2,
        )
    )
```


## Result
Based on the IaC solution the team built, the ingestion of new clients went from 2 - 3  dedicated resources over 2 - 4 weeks, to 1 dedicated resource over 1 - 2 weeks. This was essentially a 500% reduction in time and resources.

# Challenge 3
## Challenge
Reports running too long and at a high cost.

## Solution
To optimize costs and performance, I ran an analysis of running end-to-end reporting across several types of computes. To determine the types of computes to consider for each job type, I investigated the cluster CPU and memory utilization, traffic, storage, latency, and auto-scaling.

Additionally, report queries were complicated.

Example join.

## Result
Based on the job type, I then ran concurrent the same jobs for each of the potential computes. To determine different scenarios of cost and performance, I also considered conditions where the computes were scaled up or down. 

[Previous](/portfolio/timeseries/){: .btn .btn--inverse}
<!-- [Next](#link){: .btn .btn--inverse} -->