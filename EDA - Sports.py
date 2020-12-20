#!/usr/bin/env python
# coding: utf-8

# # Bhavesh Bhagat

# # Exploratory Data Analysis - Sports (IPL)

# #Task - 
# As a sports analyst, find out most succesfull teams,players and factors contributing win or loss of a team. 
# Suggest teams or players a company should endorse for its product.

# In[1]:


#importing python liberary which are used for analysing and visualizing the Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Reading Dataset from csv files as DataFrame
Matches = pd.read_csv("matches.csv")
Deliveries = pd.read_csv("deliveries.csv")


# In[3]:


Matches.head()


# In[4]:


Deliveries.head()


# In[5]:


#Basic information regading Matches DataFrame
Matches.info()


# In[6]:


#Basic information regarding Deliveries DataFrame
Deliveries.info()


# As you saw that in Matches dataset, we have given all the details of IPL matches from season 2008 to 2019.
# As you saw that in Deliveries dataset, we have given all the details of IPL matches ball by ball for all seasons.

# In[7]:


#total number of matches per season in IPL
print(Matches[['id','season']].groupby(['season']).count())


# - As per better understanding we can merge both the dataset to analysed it properly..

# In[8]:


#Merging both DataFrame to find the advanced featured for analysis
Total_data = pd.merge(Matches,Deliveries,how='inner',left_on='id',right_on='match_id')
print(Total_data.columns)


# In[9]:


#Basic view of merged DataFrame
print(Total_data.head())


# # Exploring Dataset and Creating New Features
# 
# As per our main task, we have find the most succesfull player in IPL. For that we need to calculate some advanced features like Player Batting records, Player Bowling records, Player Total Experience in IPL etc.  

# # Advance Feature For Players Batting Records
# 
# - A Player Batting records contain some important features like Total runs scored, Total Ball played, Batting Average, Strike rate, Number of Hundreds, Number of Fifties etc.
# 
# Here we calculate some advance batting features for each players..

# In[10]:


#Exploring Data to get runs as a features for each Batsman's in IPL(Per Match or Per Season) 
Batsman_data_perMatch = Total_data[['match_id','batsman','batsman_runs']]
Batsman_data_perSeason = Total_data[['season','batsman','batsman_runs']]


# In[11]:


Batsman_runs_data_perMatch = Batsman_data_perMatch.groupby(['match_id','batsman',]).sum().reset_index()
Batsman_runs_data_perSeason = Batsman_data_perSeason.groupby(['batsman','season']).sum().reset_index()


# In[12]:


#calculated batsman's runs per match or per season for futher analysis
print(Batsman_runs_data_perMatch,Batsman_runs_data_perSeason)


# In[13]:


#Exploring Data to get balls played as a features for each Batsman's in IPL(Per Match or Per Season) 
Batsman_ballData_perMatch = Total_data[(Total_data['wide_runs']==0) & (Total_data['noball_runs']==0)][['match_id','batsman','ball']]
Batsman_ballData_perSeason = Total_data[(Total_data['wide_runs']==0) & (Total_data['noball_runs']==0)][['season','batsman','ball']]


# In[14]:


Batsman_balls_data_perMatch = Batsman_ballData_perMatch.groupby(['match_id','batsman']).agg('count').reset_index()
Batsman_balls_data_perSeason = Batsman_ballData_perSeason.groupby(['season','batsman']).agg('count').reset_index()


# In[15]:


#calculated balls played by each batsman per match or per season for futher analysis
print(Batsman_balls_data_perMatch,'\n',Batsman_balls_data_perSeason)


# In[16]:


#Now finding the strike rate of batsman per match
Batsman_strike_rate_perMatch = pd.merge(Batsman_runs_data_perMatch,Batsman_balls_data_perMatch,how='inner',on=['match_id','batsman'])


# In[17]:


Batsman_strike_rate_perMatch['strike_rate'] = (Batsman_strike_rate_perMatch['batsman_runs']/Batsman_strike_rate_perMatch['ball'])*100


# In[18]:


#printing strike rate data for each batsman per match
Batsman_strike_rate_perMatch 


# In[19]:


#Data for best fielder aggregating all season's records
print(Total_data.fielder.value_counts())


# In[20]:


#merging batsman runs or balls played data per season for cal strike rate 
Batsman_strike_rate_perSeason = pd.merge(Batsman_runs_data_perSeason,Batsman_balls_data_perSeason,how='inner',on=['season','batsman'])
Batsman_strike_rate_perSeason


# In[21]:


Batsman_strike_rate_perSeason['Season_strike_rate'] = (Batsman_strike_rate_perSeason['batsman_runs']/Batsman_strike_rate_perSeason['ball'])*100


# In[22]:


#strike rate for each batsman per season
Batsman_strike_rate_perSeason


# In[23]:


#calculating number of time batsman dismissed in IPL per season
Dismissals = list()
for i in range(len(Batsman_strike_rate_perSeason)):
    Dismissals.append(Total_data[(Total_data['season']==Batsman_strike_rate_perSeason['season'][i]) & (Total_data['batsman']==Batsman_strike_rate_perSeason['batsman'][i])][['player_dismissed']].count()[0])    


# In[24]:


pd.DataFrame(Dismissals,columns=['Dismissals'])


# In[25]:


#concatenating batsman's dismissal data with strike rate data 
Batsman_Season_data = pd.concat([Batsman_strike_rate_perSeason,pd.DataFrame(Dismissals,columns=['Dismissals'])],axis=1)


# In[26]:


#because batsman average never be infinite so that each batsman has gotten out once to calculate avg.
Batsman_Season_data['Dismissals'].replace(to_replace = 0, value = 1,inplace=True)


# In[27]:


#finding batsman's average per season
Batsman_Season_data['Batsman_Average'] = Batsman_Season_data['batsman_runs']/Batsman_Season_data['Dismissals']


# In[28]:


#printing total batsman data for any player per season
Batsman_Season_data[Batsman_Season_data['batsman']=='SK Raina']


# In[29]:


#Number of time any player has dismissed in a season
Total_data[(Total_data['season']==2013) & (Total_data['batsman']=='MS Dhoni')][['player_dismissed']].count()[0]


# In[30]:


#Number of times all player dismissed in per season
Total_data[['season','player_dismissed']].groupby('season').player_dismissed.value_counts()


# In[31]:


#Top 10 batsman based on highest average in a season
Batsman_Season_data[Batsman_strike_rate_perSeason['season']==2012].nlargest(10,['Batsman_Average'])


# In[32]:


#Mean of avg of batsman's of all season to get top 15 best batsman 
Batsman_Season_data[['batsman','Batsman_Average']].groupby('batsman').agg('mean').nlargest(15,['Batsman_Average'])


# In[33]:


#calculating no of fifties or hundreds by players
No_of_Hundreds = list()
No_of_Fifty = list()
Batsman_hundreds = Total_data[['match_id','season','batsman','batsman_runs']].groupby(['match_id','batsman','season']).agg({'batsman_runs':'sum'}).reset_index()
for i in range(len(Batsman_hundreds)):
    if Batsman_hundreds['batsman_runs'][i]>=100:
        No_of_Hundreds.append(1)
    else:
        No_of_Hundreds.append(0)
    if Batsman_hundreds['batsman_runs'][i]>=50:
        No_of_Fifty.append(1)
    else:
        No_of_Fifty.append(0)


# In[34]:


print(len(No_of_Hundreds),len(No_of_Fifty))


# In[35]:


#concat fifty and hundreds of batsman's data
Batsman_fifty_hundreds = pd.concat([Batsman_hundreds,pd.DataFrame(No_of_Fifty,columns=['No_of_Fifty']),                                     pd.DataFrame(No_of_Hundreds,columns=['No_of_Hundreds'])],axis=1)
Batsman_fifty_hundreds


# In[36]:


temp_df = Batsman_fifty_hundreds.groupby(['batsman','season']).agg({'No_of_Fifty':'sum','No_of_Hundreds':'sum'}).reset_index()[['No_of_Fifty','No_of_Hundreds']]
temp_df


# In[37]:


#join batsman data of hundreds to batsman Season Data 

Batsman_Season_data = pd.concat([Batsman_Season_data,temp_df],axis=1)
Batsman_Season_data


# As the players batting records are calculated above.

# In[38]:


#Top 15 players which score most runs in IPL till 2019..

Batsman_Season_data.groupby('batsman').agg({'batsman_runs':'sum'}).reset_index().nlargest(15,['batsman_runs'])


# # Advance Feature For Players Bowling Records
# 
# - A Player Bowling records contain some important features like Total wicket taken, Bowling Economy, Bowling Average, Bowling Strike rate, Number of 4wicket haul, Number of 5wicket hual etc.
# 
# Now we have to calculated the bowling records for each players in IPL

# In[39]:


# creating data for Bolwers

Bolwers_data = Total_data[['match_id','season','bowler','over','ball','total_runs']]
Bolwers_data.groupby(['match_id','bowler']).agg({'ball':'count'}).reset_index()


# In[40]:


#Bowler data per season
Bowler_season_data = Bolwers_data.groupby(['bowler','season']).agg({'total_runs':'sum'}).reset_index()
Bowler_season_data


# In[41]:


#bowler data with total run conceded in total ball bowled in per season
Bowler_season_data = pd.concat([Bowler_season_data,Bolwers_data.groupby(['bowler','season']).agg({'ball':'count'}).reset_index()[['ball']]],axis=1)
Bowler_season_data


# In[42]:


#wicket taken by each bowler per match
Bolwers_data = Total_data[['match_id','season','bowler','over','ball','total_runs','player_dismissed']]
df_temp = Bolwers_data[['match_id','season','bowler','player_dismissed']].groupby(['match_id','season','bowler']).agg({'player_dismissed':'count'}).reset_index()
df_temp


# In[43]:


#calculating no of 4wickets or 5wicket by players
No_of_4wickets = list()
No_of_5wicket = list()

for i in range(len(df_temp)):
    if df_temp['player_dismissed'][i]>=5:
        No_of_5wicket.append(1)
    else:
        No_of_5wicket.append(0)
    if df_temp['player_dismissed'][i]==4:
        No_of_4wickets.append(1)
    else:
        No_of_4wickets.append(0)


# In[44]:


#concatenating 4wicket or 5wicket haul of each bowler per match
df_temp = pd.concat([df_temp,pd.DataFrame(No_of_4wickets,columns=['No_of_4wickets']),                      pd.DataFrame(No_of_5wicket,columns=['No_of_5wicket'])],axis=1)


# In[45]:


df_temp


# In[46]:


#bowler data of 4wicket or 5wicket haul of each bowler per season
df_temp = df_temp.groupby(['bowler','season']).agg({'No_of_4wickets':'sum','No_of_5wicket':'sum'}).reset_index()
df_temp


# In[47]:


#season wise wicket taken by bowllers
Bolwers_data[['bowler','season','player_dismissed']].groupby(['bowler','season']).agg({'player_dismissed':'count'}).reset_index()[['player_dismissed']]


# In[48]:


#season wise wicket taken by bowllers
Bowler_season_data = pd.concat([Bowler_season_data, Bolwers_data[['bowler','season','player_dismissed']].groupby(['bowler','season']).agg({'player_dismissed':'count'}).reset_index()[['player_dismissed']]],axis=1)
Bowler_season_data


# In[49]:


#calculating advance feature for bowler like Bowling_avg, Bowling Strike rate, Economy
Bowler_season_data['Bowling_avg'] = Bowler_season_data['total_runs']/Bowler_season_data['player_dismissed']
Bowler_season_data['Bowling_strike_rate'] = Bowler_season_data['ball']/Bowler_season_data['player_dismissed']
Bowler_season_data['Economy'] = Bowler_season_data['total_runs']/(Bowler_season_data['ball']/6)
Bowler_season_data = pd.concat([Bowler_season_data,df_temp[['No_of_4wickets','No_of_5wicket']]],axis=1)
Bowler_season_data


# In[50]:


#top 10 best economy of bowler
Bowler_season_data.nsmallest(10,['Economy'])


# # Player's Batting Points (PBT)
# 
# As we have batting records for each player now calculate PBT points to get the best batsman in IPL based on below formula taken by sports analyst.

# In[51]:


#calcuating player’s batting points (PBT)
#PBT = (((BattingAverage ∗0.3) + (BattingStrikeRate ∗0.4) + dN umberof H undredse+ (N umberof F ifties ∗0.2))/10)
Batsman_Season_data['PBT point'] = (((Batsman_Season_data['Batsman_Average']*0.3)+(Batsman_Season_data['Season_strike_rate']*0.4)                                    +(Batsman_Season_data['No_of_Hundreds'])+(Batsman_Season_data['No_of_Fifty']*0.2))/10) 

Batsman_Season_data


# # Player's Bowling Points (PBW)
# 
# As we have bowling records for each player now calculate PBW points to get the best bowler in IPL based on below formula taken by sports analyst.

# In[52]:


#calcuating player’s bowling points (PBW)
'''if that the bowler must have bowled minimum 100 bowls in his IPL career, then,'''
#   PBW = (((300/BowlingAverage) + (200/BowlingStrikeRate) + (300/Economy) + dN umberof4−wicketshaule ∗ 0.1 +
#            dNumberof 5−wicketshaule ∗ 0.1)/10)

Bowler_season_data_min_100_bowls = Bowler_season_data[Bowler_season_data['ball']>=100]
Bowler_season_data_min_100_bowls


# In[53]:


#PBW = (((300/BowlingAverage) + (200/BowlingStrikeRate) + (300/Economy) + dN umberof4−wicketshaule ∗ 0.1 +
#         dNumberof 5−wicketshaule ∗ 0.1)/10)
Bowler_season_data_min_100_bowls['PBW_point'] = (((300/Bowler_season_data_min_100_bowls['Bowling_avg'])+                                                  (200/Bowler_season_data_min_100_bowls['Bowling_strike_rate'])+                                                  (300/Bowler_season_data_min_100_bowls['Economy'])+                                                  (Bowler_season_data_min_100_bowls['No_of_4wickets']*0.1)+                                                  (Bowler_season_data_min_100_bowls['No_of_5wicket']*0.1) )/10)

Bowler_season_data_min_100_bowls


# In[54]:


Bowler_season_data_min_100_bowls.nlargest(10,['PBW_point'])


# # Calculating Players Experience
# 
# Adding new feature as Total Experience in Player records by finding matches played by player/total matches played as a Batsman or Bowler both.

# In[55]:


#cal. match played by player

Btm_ball_data = Total_data[(Total_data['wide_runs']==0) & (Total_data['noball_runs']==0)][['match_id','season','batsman','ball']]
total_match_played_per_season = Btm_ball_data.groupby(['season','batsman']).agg({'ball':'count','match_id':'nunique'}).reset_index() 


# In[56]:


#cal. player’s experience
#PEX = (Number of Matches Played/Total Number of Matches in IPL so far)
#Player_experienced_data = Total_data.groupby(['season','batsman']).agg({'match_id':'nunique'})

temp = Matches[['id','season']].groupby(['season']).agg({'id':'count'}).reset_index()
total_match_played_per_season = pd.merge(total_match_played_per_season, temp, how='inner',on='season')
total_match_played_per_season['bat_exp'] = total_match_played_per_season['match_id']/total_match_played_per_season['id']
total_match_played_per_season


total_match_bowled_per_season = Total_data.groupby(['season','bowler']).agg({'match_id':'nunique'}).reset_index()
total_match_bowled_per_season = pd.merge(total_match_bowled_per_season, temp,how='inner',on='season')
total_match_bowled_per_season['bowl_exp'] = total_match_bowled_per_season['match_id']/total_match_bowled_per_season['id']
total_match_bowled_per_season


# In[57]:


total_df = pd.merge(total_match_played_per_season,total_match_bowled_per_season,how='outer',left_on=['season','batsman']                    ,right_on=['season','bowler'])
total_df.fillna(0,inplace=True)
total_df


# In[58]:


#cal. Total experience of player per season

total_df['Total_exp'] = total_df['bat_exp'] + total_df['bowl_exp']
total_df


# In[59]:


total_df.nlargest(10,['Total_exp'])


# # Top Batsman in IPL

# In[60]:


#Top Batsman as per PBT points till now...
BatM_data = Batsman_Season_data[['batsman','PBT point']].groupby(['batsman'])            .agg({'PBT point':'sum'}).reset_index().nlargest(5,['PBT point'])


plt.bar(BatM_data['batsman'],BatM_data['PBT point'],width=0.25)
plt.show()


# The bar chart providing the top batsman point in IPL so far...with player name in IPL

# # Top Bowler in IPL

# In[61]:


#Top Bowler as per PBW point we calculated till now....

Bowl_data = Bowler_season_data_min_100_bowls[['bowler','PBW_point']].groupby(['bowler'])            .agg({'PBW_point':'sum'}).reset_index().nlargest(7,['PBW_point'])


plt.bar(Bowl_data['bowler'],Bowl_data['PBW_point'],width=0.25)
plt.show()


# The bar chart informs about the best bowler in IPL so far with the help of bowling point earn by players..

# # Top Experienced player in IPL

# In[62]:


#Top experienced player of IPL till now....
total_df
Exp_player = total_df[['batsman','bowler','Total_exp']].groupby(['batsman','bowler'])            .agg({'Total_exp':'sum'}).reset_index().nlargest(5,['Total_exp'])


plt.bar(Exp_player['batsman'],Exp_player['Total_exp'],width=0.25)
plt.show()


# The given chart inform about the top experienced players based on number of matches they have played in IPl so far... 

# # Exploring Most Succesful Teams
# 
# To find the most successful team in IPL till 2019, we  have calculate the Total matches played by teams in IPL/ Total Matches that they won in IPL.
# 
# - We have to calculate win percentage of each team in IPL

# In[63]:


#Total number of matches played by each teams in IPL

total_match_by_team = pd.concat([Matches['team1'],Matches['team2']],axis=0).value_counts()
print('Total matches played by each teams:\n',total_match_by_team)

#Total number of teams participated in IPL till now..

Total_teams = Matches['team1'].unique()
print('\nTotal number of teams participated in IPL till now:',len(Total_teams))


# In[64]:


#Total number of wins by each team in IPL...

pd.DataFrame(total_match_by_team,columns=['matches']).loc['Mumbai Indians'][0]
total_win_by_team = Matches['winner'].value_counts()

print('Total number of wins by each team in IPL:')
print(total_win_by_team)


# In[65]:


#Most succesfull team as per winning figures in IPL

sns.countplot(x='winner',data=Matches)
plt.show()


# The chart showing the number of wins by each team in IPL..

# # Calculating Winning % of each teams in IPL

# In[66]:


#making dict of total teams and no. of matches played by team in ipl..

TM_Played_by_team = dict()
for i in Total_teams:
    TM_Played_by_team[i] = pd.DataFrame(total_match_by_team,columns=['Total_number']).loc[i][0] 
    
print(TM_Played_by_team)


# In[67]:


#making dict of total teams and no. of matches won by team in ipl..

TM_win_by_team = dict()
for i in Total_teams:
    TM_win_by_team[i] = pd.DataFrame(total_win_by_team).loc[i][0] 
    
print(TM_win_by_team)    


# In[68]:


#Calculating Winning % of each teams in IPL

TM_win_percent = dict()
for i in Total_teams:
    TM_win_percent[i] = (TM_win_by_team[i]/TM_Played_by_team[i])*100

Team_win_percent = pd.DataFrame(TM_win_percent.items(),columns=['Teams','Win_%']).nlargest(6,['Win_%'])   

plt.bar(Team_win_percent['Teams'],Team_win_percent['Win_%'],width=0.25)
plt.show()

print('Top five teams with highest winning %:')
print(Team_win_percent)


# In[69]:


#Teams which played more than 50 matches with highest winning %

#Calculating Winning % of each teams in IPL

TM50_win_percent = dict()
for i in Total_teams:
    if TM_Played_by_team[i] >=50:
        TM50_win_percent[i] = (TM_win_by_team[i]/TM_Played_by_team[i])*100

Team_win_percent = pd.DataFrame(TM50_win_percent.items(),columns=['Teams','Win_%']).nlargest(6,['Win_%'])   

plt.bar(Team_win_percent['Teams'],Team_win_percent['Win_%'],width=0.25)
plt.show()

print('Top five teams played more than 50 matches in ipl with highest winning %:')
print(Team_win_percent)


# As we saw that Chennai Super Kings and Mumbai Indians are top 2 teams which played more than 50 matches in IPL till 2019 and have highest winning percentage of 60.97% and 58.28%.

# # Top Battle between any two teams

# In[70]:


#Top 10 battle win b/w two teams

Matches.groupby(['team1','team2','winner']).agg({'winner':'count'}).nlargest(10,['winner'])


# # Basic Understanding regarding players

# In[71]:


#Top batsman and bowler according to their earn points in IPL...

sns.lineplot(x="batsman", y="PBT point", data=BatM_data) 
plt.show()

sns.lineplot(x="bowler", y="PBW_point", data=Bowl_data) 
plt.show()


# In[72]:


#Top 5 Batsman with highest batting point in IPL...

for i in range(len(BatM_data)):
    print('PBT point of',BatM_data.reset_index()['batsman'][i])
    plt.pie(Batsman_Season_data[Batsman_Season_data['batsman']==BatM_data.reset_index()['batsman'][i]]['PBT point'],            labels=Batsman_Season_data[Batsman_Season_data['batsman']==BatM_data.reset_index()['batsman'][i]]['season'],            startangle=90,autopct = '%1.3f')
    plt.legend()
    plt.show()


# # Finding best players in IPL using their MVP point
# 
# MVP(Most valuable player) point is calculated based on the performance point(PBT,PBW or PEX) earn by player in IPL...

# In[73]:


#merging player batting and bowling data to calculate MVP point..

Player_total_details = pd.merge(pd.merge(Batsman_Season_data, Bowler_season_data_min_100_bowls, how='outer',left_on='batsman',right_on='bowler'),         total_df,how='outer',left_on='batsman',right_on='batsman') 


# In[74]:


#Basic Infomation of player full data
Player_total_details.info()


# In[75]:


#calculating each players MVP point..

Player_total_details['MVP_point'] = Player_total_details['PBT point']-Player_total_details['PBW_point']+Player_total_details['Total_exp']
Player_total_details.fillna(0,inplace=True) 
Player_total_details


# In[76]:


#top 10 players as per batting or bowling point experienced in IPL till now...

Top_10_Players = Player_total_details.groupby(['batsman','bowler_x']).agg({'MVP_point':'sum'}).reset_index().nlargest(10,['MVP_point'])
Top_10_Players.rename(columns={'batsman':'Players'},inplace=True)


plt.bar(Top_10_Players['Players'],Top_10_Players['MVP_point'],width=0.10)
plt.show()
Top_10_Players[['Players','MVP_point']]


# As we saw ealier we found the best batsman or bowler but using the MVP points we got the information of best player in IPL which contributing in the teams performance for winning or losing any match in IPL.
# 
# As we saw two most succesful teams (CSK or MI), In top 10 best player most of player are from those 2 teams for which they contributing to their teams to win or lose any match. 
