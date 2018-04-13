# Table Information

Our two tables are from stage 2 and the two tables are book information about
 data science books in both the library of UW-Madison ([University of 
 Wisconsin–Madison](https://www.wisc.edu/)) and UIUC ([University of Illinois Urbana-Champaign](http://illinois.edu/)).
 
`TableA.csv` is about UW-Madison and `TableB.csv` is about UIUC.

# The meaning of the attributes

There are 7 attributes of books contained in each table

1. Title: book title, string
2. Author: book authors, string
3. Publication:information of publishers, string
4. Format: book types, category, e.g. journal, magazines
5. ISBN: Search Results International Standard Book Number,integer
6. Series: book series information, string
7. Physical Details:Physical description in book cataloging, string

With one extra column about the ID in each table to indict the unique records
 of books in that table.

# Number of Tuples per table

There are :

* 6963 tuples in table `TableA.csv`
* 5730 tuples in table `TableB.csv`

# Other Data Files in In tis directory

## file that lists all tuple pairs that survive the blocking step. 

This is the  file [Set_C.csv]() and there are `1286` tuples survive the 
blocking 
steps.

## file that lists all tuple pairs in the sample you have taken

This is the file [Set_G.csv]() which contains `500` tuple pairs and each with a
 label to indicate whether they are the same entity. 
  

## file that describe the sets I 

## file that describe the sets J