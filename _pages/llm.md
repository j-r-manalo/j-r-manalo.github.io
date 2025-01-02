---
permalink: /portfolio/llm/
title: "Boosting Web Scraping Accuracy with LLM-Powered Pipelines"
# excerpt: ""
header:
  overlay_image: /assets/images/david-clode-CKCpKebng_I-unsplash.jpg  
  caption: "Photo credit: [David Clode](https://unsplash.com/@davidclode) on [Unsplash](https://unsplash.com)"
last_modified_at: 2024-12-30T11:59:26-04:00
author_profile: true
layout: single
intro: 
  - excerpt: 'A showcase of how optimizing infrastructure and implementing strategic prompt engineering significantly improved a web scraping pipeline that leveraged LLMs.'
toc: true
toc_label: "Table of Contents"
toc_sticky: true
---
{% include feature_row id="intro" type="center" %}

# Background
Our company provided business intelligence by scraping news, events, and job postings, enabling clients to better 
understand their community, ideas, and resources. However, the existing scraping code often struggled with dynamically generated content, inconsistent website structures (e.g., varying HTML layouts, frequently changing class names, and the use of iframes or dynamic loading techniques), and struggling to extract relevant information from unstructured web data due to a lack of contextual understanding. This led to inaccurate datasets and weeks of manual QA. To address these challenges, I led an initiative that focused on optimizing both the underlying infrastructure and the prompt engineering strategies. This resulted in a >400% improvement in model accuracy, significantly expanding its potential and enabling automated production deployment.

# Technical Approach

## Infrastructure Improvements
### Reducing Data Loss from Slow-Loading Web Pages by 90% Through Timeout Optimization
#### Challenge
One of the main issues with our web scraping pipeline was data loss due to slow-loading dynamic pages. Our original implementation used a timeout rule to prevent indefinite hangs, but this resulted in significant data loss when pages loaded after the timeout period.

#### Solution
Through analysis of page load times across a sample of problematic URLs using browser developer tools, I determined 
that increasing the timeout by 10 seconds significantly improved scraping success. This optimization addressed the 
issue for 90% of these pages. Below is a sample of the code the team developed and implemented that demonstrates the
timeout 
strategy:

**Code Sample of the Time-Out Rule Implementation**
```js
      // Set timeout promise.
      const timeoutPromise = new Promise((resolve) => {
        const timeout = 10000; // Increased timeout to 10 seconds.
        setTimeout(() => resolve('timed out'), timeout);
      });

      // Implement timeout using Promise.race to resolve with either page load or timeout.
      const page_load_result = await Promise.race([
        page.goto(url, {
          // Wait for DOM content and network to be idle.
          waitUntil: [
            "domcontentloaded",
            "networkidle0"
          ],
        }),
        timeoutPromise
      ]);

      // Check for timeout.
      if (page_load_result === 'timed out') {
        throw new Error(`page.goto timed out after 10s ... ${url}`);
      }
  
      // Introduce an additional 10-second delay for extra safety.
      await new Promise(function(resolve) {
        setTimeout(resolve, (10 * 1000));
      });
```

#### Result
This change resulted in a 90% reduction in data loss from time-outs. This significantly improved the completeness and reliability of our datasets.

### Improving Web Scraping Efficiency and Data Quality through Targeted Crawling
#### Challenge
Our web scraping pipeline exhaustively crawled pages starting from a set of initial URLs, resulting in a large 
volume of irrelevant data and necessitating several days of QA per client to ensure high data quality. This spurious 
data impacted the reliability of our insights and significantly increased operational overhead.

#### Solution
I facilitated discussions with stakeholders across QA, data engineering, and client teams to define a more efficient crawling approach. We identified that limiting the crawl depth would significantly improve data relevance. The new approach extracts all links from the initial client-provided URL and then crawls only those linked pages, limiting the crawl depth to one additional level.

#### Result
Testing on a representative sample of client websites showed that this approach captured 95% of relevant links, as 
defined by links to client events, news, and jobs. While this approach captured 95% of relevant links, we acknowledged a potential for missing some less common link structures. This trade-off prioritized data quality and reduced QA effort. This change enabled automated production deployment, eliminating weeks of manual QA and significantly improving efficiency.

## Prompt Engineering Strategies
### Content Understanding and Categorization
#### Challenge
In our web scraping pipeline, we initially relied on a two-step LLM process for content categorization and extraction. First, we used a prompt to classify web pages as _news_, _event_, or _job_. Then, if a page was classified into one of these categories, a second prompt was used to extract relevant information (e.g., job titles, dates, locations). This approach yielded a low accuracy rate of approximately 45% for the initial categorization, leading to significant data loss.

**Original Prompting Approach (Example)**

- **Classification Prompt**: "Classify the following web page content as 'news', 'event', or 'job': [web page content]"
- **Extraction Prompt** (if classified as "job"): "Extract the job title, company, location, and description from the 
following job posting: [web page content]"

This two-step process was inefficient and error-prone. The classification prompt often miscategorized pages, leading to the extraction prompt being applied incorrectly or not at all.

#### Solution
**Contextual Pre-Labeling and Streamlined Extraction**

After collaborating with stakeholders across QA, data engineering, and client teams, we identified a key insight: the context of the initial seed URLs provided by clients could be leveraged to significantly improve accuracy. We realized that links found on a client-provided page were highly likely to belong to the same category as the seed page itself.

This insight allowed us to shift from a classification-first approach to a direct extraction approach with contextual pre-labeling. Instead of asking the LLM to classify the content, we pre-labeled the content based on the client's initial URL category. This allowed us to use a more focused and effective extraction prompt.

**Revised Prompting Approach (Example)**

 - **Extraction Prompt (with Pre-labeling)**: "The following content is from a job board. Extract the job title, 
 company, location, and description: [web page content]"

By providing the LLM with the context upfront ("The following content is from a job board"), we eliminated the 
ambiguity that caused the low initial accuracy. (Note, this is a form of context injection in prompt engineering.) We 
also 
eliminated a step, making the pipeline more efficient.

#### Result
This change in strategy, specifically the introduction of contextual pre-labeling in the prompts, dramatically 
improved accuracy rates to nearly 98%. This represented a significant improvement in data capture and pipeline efficiency. This demonstrates the power of using contextual information to guide LLM behavior and improve performance.

# Overall Results and Analysis
Overall, the combined impact of infrastructure enhancements and strategic prompt engineering resulted in a >400% 
improvement in LLM accuracy. This dramatic increase enabled us to automate over 80% of previously manual data 
analysis tasks, leading to substantial cost savings and a significant boost in team productivity. The success of 
this initiative was recognized company-wide, and the improved web-scraping and LLM is now a core component of our data 
analytics 
platform.


[Previous](/portfolio/datawarehouse/){: .btn .btn--inverse}