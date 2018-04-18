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

This is the  file [Set_C.csv](Set_C.csv) and there are `606` tuples survive
 the 
blocking 
steps.

## file that lists all tuple pairs in the sample you have taken

This is the file [Set_G.csv](./Set_G.csv) sampled from [Set_C.csv](Set_C.csv)  which contains `300` 
tuple pairs
 and
 each with a
 label to indicate whether they are the same entity. 
  

## file that describe the sets I

We split [Set_G.csv](./Set_G.csv) into [Set_I](./Set_I.csv) as training set
which contains `200` tuple pairs.

## file that describe the sets J

This is the reaming tuple pairs from [Set_G.csv](./Set_G.csv) and not in [Set_I](./Set_I.csv)
which contains `100` tuple pairs as test set.

## other files are meta-data that used during training

These files are `label_S_ltable.csv`,
`label_S_ltable.metadata`,
`label_S_rtable.csv`,	 
`label_S_rtable.metadata`.

