# Know Their Names Project Overview

## Main Areas of Concern

1.  Problem
2.  Theory
3.  Data
4.  Code

## Problem

From a technical perspective, this project is concerned with (1) reducing a set of heterogeneous data records about enslaved people to a canonical data set of unique individuals and their personal attributes, and (2) creating a network connecting these individuals by kinship and other forms of social relationship.

The general problem domain is called "entity resolution," the idea being that your trying to resolve the identities of people from incomplete and fragmented data sets.

> Note that the problem domain goes by many names – or, rather, involves many closely related and overlapping domain, such as "record linking" and "data linking." See Rachel House's video below for a discussion of this.

The specific problem in this project is the disconnect between records taken before and after 1870, the first data of the US Census after the civil war. Records of enslaved people before this data are notoriously incomplete; for example, they do not list last names.

## Theory

The quickest way to get acquainted with the general problem domain of entity resolution of this project is to watch these videos. The first two are each less than two minutes and given a good overview of the nature of the problem and the pipeline used to approach it.

[Record Linkage Explained](https://youtu.be/WbjC4ikuWi4?si=-FDmn3QAEb-taoAy) (1m 26s)

[What is Entity Resolution?](https://youtu.be/MvuG2herj6s?si=R_2iayfT7D6XSoDl) (1m 37s)

The third video is a talking given by Rachel House of S&P Global at the 2021 Women in Data Science conference sponsored by the School of Data Science.

[An Introduction to Data Linking](https://youtu.be/Vwr_RowGSd8?si=O_Uv5uP82S7EKyzq) (30m 07s).

For a deeper dive, see the t[he articles shared by the client](https://drive.google.com/drive/folders/1Ggb284c_VLFtK7ZEfhx1lJVgSocaeiXB?usp=drive_link).

## Data

Although the project as a whole involves many data sets, for now your main concerns is with the MENTIONS and ASSERTIONS data. These have already been normalized to a large degree for you.

## Code

Although there are many tools you. can bring to bear on the problem, a good place to start is with [Python Record Linkage Toolkit.](https://recordlinkage.readthedocs.io/en/latest/index.html) It works with Pandas and is suited for projects of this scale.