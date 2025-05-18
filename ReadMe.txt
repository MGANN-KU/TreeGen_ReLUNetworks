
File: Finding_outward_edges.py
This file contains an implementation of the proposed generative ReLU to find the indices and labels of 
outward edges of the given inward edges in the Euler string.

Input:
- t: Input Euler string of length 2n
- m: Size of the symbol set
- x: String of length d to identify inward edges

Output:
- y: The outward edges of t following x

Example:
t = [3, 2, 7, 2, 4, 9, 7, 4, 9, 8]
d = 3
m = 5
x = [1, 3, 0]
y = [10, 7, 0]

--------------------------------------------------

File: TS_d.py
This file contains an implementation of the TS_d-generative ReLU to generate Euler strings with a given edit 
distance due to substitution.

Input:
- t: Input Euler string of length 2n
- d: Edit distance
- m: Size of the symbol set
- x: String of length 2d to identify substitution operations

Output:
- u: Euler string obtained by applying the substitution operation on t following x with distance at most 2d

Example:
t = [3, 2, 7, 2, 4, 9, 7, 4, 9, 8]
d = 3
m = 5
x = [1, 3, 1, 5, 1, 2]
u = [5, 2, 7, 1, 4, 9, 6, 4, 9, 10]

--------------------------------------------------

File: TD_d.py
This file contains an implementation of the TD_d-generative ReLU to generate Euler strings with a given edit
 distance due to deletion only.

Input:
- t: Input Euler string of length 2n
- d: Edit distance
- m: Size of the symbol set
- x: String of length d to identify deletions

Output:
- u: Euler string obtained by applying the deletion operation on t following x with distance 2d

Example:
t = [3, 2, 7, 2, 4, 9, 7, 4, 9, 8]
d = 3
m = 5
x = [1, 3, 1]
u = [2, 7, 4, 9, 4, 9]

--------------------------------------------------

File: TI_d.py
This file contains an implementation of the TI_d-generative ReLU to generate strings with a given edit distance
 due to insertion only.

Input:
- t: Input Euler string of length 2n
- d: Edit distance
- m: Size of the symbol set
- x: String of length 4d to identify insertions

Output:
- u: String obtained by applying the insertion operation on t following x with distance 2d

Example:
t = [3, 2, 7, 2, 4, 9, 7, 4, 9, 8]
d = 4
m = 5
x = [1, 0, 3, 0, 2, 4, 1, 1, 3, 2, 5, 1, 4, 1, 3, 5]
u = [1, 6, 5, 3, 2, 7, 4, 2, 4, 9, 3, 8, 7, 4, 9, 9, 8, 10]

--------------------------------------------------

File: TE_d_unified.py
This file contains an implementation of the TE_d-generative ReLU to generate strings with a given edit distance
 due to substitution, deletion, and insertion operations simultaneously.

Input:
- t: Input Euler string of length 2n
- d: Edit distance
- m: Size of the symbol set
- Delta: Small threshold value
- x: String of length 7d to identify all edit operations

Output:
- u: String obtained by applying deletion, substitution, and insertion on t following x with distance at most 2d

Example:
t = [3, 2, 12, 2, 4, 14, 12, 4, 14, 13]
d = 3
m = 10
Delta = 0.01
x = [0.3, 0, 0.38, 0, 0.46, 0.55, 0, 0.6, 0.88, 0.66, 0.75, 0, 0.55, 0.87, 0.03, 0.02, 0.45, 0.09, 0, 0.7, 0.5]
u = [5, 3, 2, 6, 16, 12, 4, 14, 13, 15]
