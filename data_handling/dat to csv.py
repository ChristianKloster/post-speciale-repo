import csv

# read flash.dat to a list of lists
datContent = [i.split('::') for i in open("../data/ML 10m/ratings.dat").readlines()]

# write it as a new CSV file
with open("../data/ML 10m/ratings.csv", "w") as f:
    for n in datContent:
        f.write(n[0])
        f.write(", ")
        f.write(n[1])
        f.write(", ")
        f.write(n[2])
        f.write("\n")

