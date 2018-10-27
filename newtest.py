import MySQLdb

import time

import numpy as np

import csv

import math

#from scipy.sparse import coo_matrix

db = MySQLdb.connect("localhost", "root", "123456", "twitter1", charset='utf8' )

cursor = db.cursor()



'''
with open("user.csv", 'r') as in_csv:

    tmp = csv.reader(in_csv)

    column1 = [row for row in tmp]

user_id = [i[0] for i in column1]

user_count = [i[1] for i in column1]

user_following = [i[2] for i in column1]

user_follower = [i[3] for i in column1]

user_friends = [i[4] for i in column1]

user_favourite = [i[5] for i in column1]





with open("status.csv", 'r') as in_csv:

    tmp = csv.reader(in_csv)

    column2 = [row for row in tmp]

status_id = [i[0] for i in column2]

status_user_id = [i[1] for i in column2]

status_time = [i[2] for i in column2]

status_reply = [i[3] for i in column2]

status_retweet = [i[4] for i in column2]

status_favourite = [i[5] for i in column2]





with open("retweet.csv", 'r') as in_csv:

    tmp = csv.reader(in_csv)

    column3 = [row for row in tmp]

retweet_status_id = [i[0] for i in column3]

retweet_user = [i[1] for i in column3]

retweet_r_status_id = [i[2] for i in column3]

retweet_r_user = [i[3] for i in column3]

retweet_follower = [i[4] for i in column3]

retweet_time = [i[5] for i in column3]

retweet_retweettime = [i[6] for i in column3]

#print(user_id)

'''

#print("status done...")



#fout.close()

#

#sql = "SELECT * FROM twitter1.retweet"

'''cursor.execute(sql)

results = cursor.fetchall()

retweet_status_id = []

retweet_user = []

retweet_r_status_id = []

retweet_r_user = []

retweet_follower = []

retweet_time = []

retweet_retweettime = []

fout = open("retweet-test.csv", "w")'''

#

'''i = 0

for row in results:

    #print("in")

    if (i < 500):

        i = i + 1

        retweet_status_id.append(row[0])

        retweet_user.append(row[1])

        retweet_r_status_id.append(row[2])

        retweet_r_user.append(row[3])

        retweet_follower.append(row[4])

        retweet_time.append(time.mktime(time.strptime(str(row[5]), "%Y-%m-%d %H:%M:%S")))

        retweet_retweettime.append(time.mktime(time.strptime(str(row[6]), "%Y-%m-%d %H:%M:%S")))

        fout.write(str(row[0]) + "," + str(row[1]) + "," +str(row[2]) + "," +  str(row[3]) + "," + str(row[4]) + ","+str(time.mktime(time.strptime(str(row[5]), "%Y-%m-%d %H:%M:%S"))))

        fout.write("," + str(time.mktime(time.strptime(str(row[6]), "%Y-%m-%d %H:%M:%S"))) + "\n")'''



#print("retweet done...")
'''
sql = "SELECT * FROM twitter1.user"



cursor.execute(sql)

results = cursor.fetchall()

user_id = []

user_count = []

user_following = []

user_follower = []

user_friends = []

user_favourite = []

fout = open("user.csv", "w")

for row in results:

    user_id.append(row[1])

    user_count.append(row[2])

    user_following.append(row[3])

    user_follower.append(row[4])

    user_friends.append(row[5])

    user_favourite.append(row[6])

    fout.write(str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "\n")
fout.close()
'''
sql = "SELECT * FROM twitter1.status"

cursor.execute(sql)

results = cursor.fetchall()

status_id_tmp = []

status_user_id_tmp = []

status_time_tmp = []

#status_reply = []

#status_retweet = []

#status_favourite = []

#fout = open("status.csv", "w")

for row in results:

    if (row[3] != None):



        status_id_tmp.append(row[0])

        status_user_id_tmp.append(row[1])

              #print(row)

        status_time_tmp.append(time.mktime(time.strptime(str(row[3]), "%Y-%m-%d %H:%M:%S")))

       # status_reply.append(row[4])

       # status_retweet.append(row[5])

       # status_favourite.append(row[6])

       # fout.write(str(row[0]) + "," + str(row[1]) + "," + str(time.mktime(time.strptime(str(row[3]), "%Y-%m-%d %H:%M:%S"))))

       # fout.write( "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "\n")

#fout.close()
'''sql = "SELECT * FROM twitter1.user"



cursor.execute(sql)

results = cursor.fetchall()

user_id = []

user_count = []

user_following = []

user_follower = []

user_friends = []

user_favourite = []

fout = open("user.csv", "w")

for row in results:
    if (row[1] in status_user_id):
        user_id.append(row[1])

        user_count.append(row[2])

        user_following.append(row[3])

        user_follower.append(row[4])

        user_friends.append(row[5])

        user_favourite.append(row[6])

        fout.write(str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "\n")

fout.close()
#print("user done...")

#print(len(user_id))
'''
sql = "SELECT * FROM twitter1.retweet"



cursor.execute(sql)

results = cursor.fetchall()

retweet_status_id = []

retweet_user = []

retweet_r_status_id = []

retweet_r_user = []

retweet_follower = []

retweet_time = []

retweet_retweettime = []

fout = open("retweet.csv", "w")

#

for row in results:

    #print("in")

    if (row[0] in status_id_tmp):

        #i = i + 1

        retweet_status_id.append(row[0])

        retweet_user.append(row[1])

        retweet_r_status_id.append(row[2])

        retweet_r_user.append(row[3])

        retweet_follower.append(row[4])

        retweet_time.append(time.mktime(time.strptime(str(row[5]), "%Y-%m-%d %H:%M:%S")))

        retweet_retweettime.append(time.mktime(time.strptime(str(row[6]), "%Y-%m-%d %H:%M:%S")))

        fout.write(str(row[0]) + "," + str(row[1]) + "," +str(row[2]) + "," +  str(row[3]) + "," + str(row[4]) + ","+str(time.mktime(time.strptime(str(row[5]), "%Y-%m-%d %H:%M:%S"))))

        fout.write("," + str(time.mktime(time.strptime(str(row[6]), "%Y-%m-%d %H:%M:%S"))) + "\n")

fout.close()
'''
#print(i)

#print("retweet done...")

#sql = "SELECT * FROM twitter1.user"

#

#print('9989862' in retweet_user)

'''
'''cursor.execute(sql)

results = cursor.fetchall()

user_id = []

user_count = []

user_following = []

user_follower = []

user_friends = []

user_favourite = []

fout = open("user-test.csv", "w")

for row in results:

    if ((row[1] in retweet_user or row[1] in retweet_r_user) and row[1] in status_user_id):

      user_id.append(row[1])

      user_count.append(row[2])

      user_following.append(row[3])

      user_follower.append(row[4])

      user_friends.append(row[5])

      user_favourite.append(row[6])

      fout.write(str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "\n")

print("user done...")'''
#fout.close()

#sql = "SELECT * FROM twitter1.status"

'''status_id = []

status_user_id = []

status_time = []

status_reply = []

status_retweet = []

status_favourite = []

cursor.execute(sql)

results = cursor.fetchall()

fout = open("status-test.csv", "w")

for row in results:

    if (row[3] != None and row[0] in retweet_status_id and row[1] in user_id):



        status_id.append(row[0])

        status_user_id.append(row[1])

              #print(row)

        status_time.append(time.mktime(time.strptime(str(row[3]), "%Y-%m-%d %H:%M:%S")))

        status_reply.append(row[4])

        status_retweet.append(row[5])

        status_favourite.append(row[6])

        fout.write(str(row[0]) + "," + str(row[1]) + "," + str(time.mktime(time.strptime(str(row[3]), "%Y-%m-%d %H:%M:%S"))))

        fout.write( "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "\n")'''
print(len(retweet_status_id))
status_id = []
user_id = []
status_time = []
status_user_id = []
for i in range(len(retweet_status_id)):
    if (retweet_status_id[i] not in status_id):
        status_id.append(retweet_status_id[i])
       # index = status_id_tmp.index(id)
        status_user_id.append(retweet_user[i])
        status_time.append(retweet_time[i])
for id in retweet_user:
    if (id not in user_id):
        user_id.append(id)

for id in retweet_r_user:
    if (id not in user_id):
        user_id.append(id)

fout = open("user_id.csv", "w")
for id in user_id:
    fout.write(str(id) + "\n")
fout.close()
fout = open("status_id.csv", "w")
print(len(status_id))
for i in range(len(status_id)):
    fout.write(str(status_id[i]) + "," + str(status_user_id[i]) + "," + str(status_time[i])+ "\n")

print("Data done..")
db.close()

M = len(user_id)

N = len(status_id)

#print(M)

#print(N)
nn = 0
#print(len(retweet_status_id))

R = np.zeros((M, N), np.float16)
T = np.zeros((M, M), np.float16)
RR = np.zeros((N, M), np.float16)
H = np.zeros((M, M), np.float16)
nn = 0

#print(len(retweet_status_id))

for i in range(len(retweet_status_id)):

    index_u = user_id.index(retweet_r_user[i])

    index_m = status_id.index(retweet_status_id[i])

    index_v = user_id.index(retweet_user[i])

    #print(str(index_u) + " " + str(index_m))

    R[index_u,  index_m] = 1

    nn = nn + 1

    RR[index_m, index_v] = 1
    T[index_u, index_v] = T[index_u, index_v] + 1
#print(nn)

TA = np.load("lda.npy")

#print(TA)

print("TA done...")

D = [[] for i in range(M)]

for i in range(len(status_user_id)):

    index = user_id.index(status_user_id[i])

    D[index].append(i)

#print(user_id[M-1])

(TA_M, TA_N) = TA.shape

II = np.zeros((M, TA_N), np.float16)
print(M)
print(len(status_id))
print(TA.shape)
print(len(status_user_id))
#print(D)

for i in range(M):

    for j in range(len(D[i])):
	
        II[i,:] = II[i,:] + TA[D[i][j], :]

        #print(II[i,:])

    if (len(D[i]) > 0):

        II[i,:] = II[i,:]*(1/len(D[i]))

print("II done...")

#print(II)
Interest_mess = np.load("interest_mess.npy")
Interest_user = np.load("interest_user.npy")
'''Interest_mess = np.zeros((M, N), np.float16)

Interest_user = np.zeros((M, M), np.float16)
for i in range(M):

    for j in range(N):

        if (j % 10000 == 0):

            print("mess " + str(i) + " " + str(j))

        Interest_mess[i, j] = II[i, :].dot(TA[j, :])/((np.sqrt(II[i, :].dot(II[i, :]))) * (np.sqrt(TA[j, :].dot(TA[j, :]))))
np.save("interest_mess.npy",Interest_mess)
'''
print("Interest_mess done..")
'''for i in range(M):

    for j in range(M):

        if (j % 10000 == 0):

            print("user " + str(i) + " " + str(j))



        Interest_user[i, j] = II[i, :].dot(II[j, :])/((np.sqrt(II[i, :].dot(II[i, :]))) * (np.sqrt(II[j, :].dot(II[j, :]))))
'''
#print(Interest_user)
#np.save("interest_user.npy",Interest_user)
print("Interest done...")

#T = R.dot(RR)
'''T = np.zeros((M, M),np.float16)
for i in range(M):

    for j in range(M):

        if (j % 100 == 0):

            print("T " + str(i) + " " + str(j))

        for k in range(N):

            T[i,j] = T[i, j] + R[i, k]*RR[k, j]

'''
print("T-first done..")
for i in range(M):

    for j in range(M):

        if (i == j):

            T[i, i] = len(D[i])
print("T done...")
P = np.zeros((M, M), np.float16)

H = np.zeros((M, M), np.float16)
print("P,H done...")
tm = status_time
print("tm done..")
'''timeslot = np.zeros((M, M, 2), np.float16)
print("timeslot founded..")
for i in range(M):
    for j in range(N):
        if (R[i,j] == 1):

            
            print("time " + str(i) + " " + str(j))
            v = user_id.index(status_user_id[j])
            if (float(timeslot[i, v, 0]) > float(status_time[j])):
                timeslot[i, v, 0] = status_time[j]
            elif (float(timeslot[i, v, 1]) < float(status_time[j])):
                timeslot[i, v, 1] = status_time[j]
np.save("timeslot.npy", timeslot)'''
timeslot = np.load("timeslot.npy")
print("timeslot done...")
#n = np.zeros((M, M), np.float16)
n = np.load("n.npy")
'''for i in range(M):

    for j in range(M):

        n[i, j] = T[i, j] / (timeslot[i, j, 1] - timeslot[i, j, 0])
np.save("n.npy", n)'''
print("n done..")
P = np.load("P.npy")
'''for i in range(M):

    for j in range(M):

        P[i, j] = T[i, j]/T[j, j] * Interest_user[i, j]
np.save("P.npy", P)'''
print("P done..")
#print(H.shape)
print(len(retweet_user)) 
#print(timeslot.shape)
'''for i in range(len(retweet_r_user)):
    if (i % 100 == 0):
        print("retweet " + str(i))
    u = user_id.index(retweet_r_user[i])
    v = user_id.index(retweet_user[i])
    H[u, v] = n[u, v]/(timeslot[u, v, 0] - float(retweet_time[i]))
for i in range(M):
    for j in range(M):
        H[i,j] = H[i, j] / T[i, j]'''
#np.save("H.npy", H)
H = np.load("H.npy")
print("n P H done...")
w = 0.5
'''Influence = np.zeros((M, M), np.float16)
for i in range(M):
    for j in range(M):
        #print(v)
        Influence[i, j] = w*P[i, j] + (1-w)*H[i, j]
np.save("Influence.npy", Influence)'''
Influence = np.load("Influence.npy")
k = 0.5
print("Influence done...")
#S = np.zeros((M, N), np.float16)
S_final = np.zeros((M, N), np.float16)
'''for i in range(M):
    for j in range(N):
        if (R[i, j] == 1):
            v = user_id.index(status_user_id[j])
            #print(Influence[i, v])

            S[i, j] = k * Interest_mess[i, j] + (1-k) * Influence[i, v]
np.save("S.npy", S)'''
S = np.load("S.npy")
print("S done...")
#print(D)

#print(S)

for i in range(M):

    for j in range(N):

        if (R[i, j] == 0):

            S_final[i, j] = S[i, j]
            if (S[i, j] == 0):
                S_final[i,j] = np.random.uniform(0, 0.0001)

        else:

            S_final[i, j] = 1
np.save("S_final.npy", S_final)
print("S_final done...")

#print(S)

print(S_final)

R0 = 1

R1 = 0

lam = 0.5

#K = round((M + N) / 2)

K = 1

U = np.random.random(size = (M, K))

V = np.random.random(size = (N, K))

#print(S_final[0,:])

eyes = np.eye(K)

#while ((R1-R0)*(R1-R0) > 0.001):
for p in range(100):
    R0 = R1
    print(p)
    for i in range(M):

        #print(V.shape)

        #print(np.diag(S_final[i,:]).shape)

        #print(R[i,:].shape)

        #print(S_final[i,:].sum())
	#if (i % 1000 == 0):
         #   print("U " + str(i))

        op1 = R[i,:].dot(np.diag(S_final[i,:])).dot(V)#np.dot(np.dot(R[i,:], np.diag(S[i,:]), V)

        op2 = V.T.dot(np.diag(S_final[i,:])).dot( V)

        op3 = lam * S_final[i,:].sum() * eyes

        #print(op1.shape)

        #print(op2.shape)

        #print(op3.shape)

        #print(S_final[i,:])

        U[i,:] = np.dot(op1, np.linalg.inv(op2 + op3))

        #U[i] = np.linalg.inv(op3)



    for j in range(N):
	#if (j % 1000 == 0):
         #   print("V " + str(j))

        op1 = R.T[j,:].dot(np.diag(S_final[:,j])).dot(U)#np.dot(np.dot(R[i,:], np.diag(S[i,:]), V)

        op2 = U.T.dot(np.diag(S_final[j,:])).dot(U)

        op3 = lam * S_final[j,:].sum() * eyes

        V[i,:] = np.dot(op1, np.linalg.inv(op2 + op3))

    R1 = 0

    for i in range(M):

        for j in range(N):

            R1 = R1 + S_final[i, j] * ((R[i, j] - U[i,:] * V[j,:])*(R[i, j] - U[i,:] * V[j,:]) + lam * ((np.linalg.norm(U,ord=2))*(np.linalg.norm(U,ord=2)) + (np.linalg.norm(V,ord=2))*(np.linalg.norm(V,ord=2))))

            #print(S_final[i, j] * ((R[i, j] - U[i] * V[j])*(R[i, j] - U[i] * V[j])))

    print((R0-R1)*(R0 - R1))

    #print(R1)
np.save("U.npy", U)
np.save("V.npy", V)
print("calculate done...")

ret = np.zeros((M, N), np.float16)

for i in range(M):

    for j in range(N):

        ret[i, j] = U[i] * V[j]

print("ret done...")

np.save("result.npy",ret)
print("done!")
