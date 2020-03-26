import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns


titanic=pd.read_csv('train.csv')
titanic.head() #pandas indexed it by default
titanic.info() #gives info about the data in all the columns, if there are any non null values
'''PART01'''
'''1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
2.) What deck were the passengers on and how does that relate to their class?
3.) Where did the passengers come from?
4.) Who was alone and who was with family?'''

sns.catplot(x='Sex',data=titanic,kind='count')#should work but is not showing so i used histogram plot instead catplot is substitute for factor plot
sns.catplot(x='Sex',y='Survived',hue='Pclass',data=titanic,kind='bar')
plt.show()
def male_female_child(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex

titanic['person']=titanic[['Age','Sex']].apply(male_female_child,axis=1)
titanic.head()
sns.catplot(x='Pclass',y='Survived',hue='person',data=titanic,kind='bar')
titanic['Age'].hist(bins=70)
titanic['Age'].mean()
titanic['person'].value_counts()

'''PART2'''
#KDE plots with facit grid
fig=sns.FacetGrid(titanic,hue='Sex',aspect=3)#aspect is how big or long you want your plot to be, this line only is to make a grid for plot
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic['Age'].max() #we use this variable to set limit to the plot
fig.set(xlim=(0,oldest))#limitting the plot to +ve values
fig.add_legend()#adds legend


 #TO INCLUDE CHILDREN(even though there is no overlapping data, plots overlap because of KDE pot band width)
fig=sns.FacetGrid(titanic,hue='person',aspect=3)#aspect is how big or long you want your plot to be, this line only is to make a grid for plot
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic['Age'].max() #we use this variable to set limit to the plot
fig.set(xlim=(0,oldest))#limitting the plot to +ve values
fig.add_legend()#adds legend

#for class
fig=sns.FacetGrid(titanic,hue='Pclass',aspect=3)#aspect is how big or long you want your plot to be, this line only is to make a grid for plot
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic['Age'].max() #we use this variable to set limit to the plot
fig.set(xlim=(0,oldest))#limitting the plot to +ve values
fig.add_legend()#adds legend


'''2.) What deck were the passengers on and how does that relate to their class?'''
titanic.head()
deck=titanic['Cabin'].dropna()
print(deck)#gives cabin deck and number details form data where nan is dropped say C85 cabin C and number 85's nan is dropped
'''we only need deck details but not number from deck''' #lets make a list
List=[]
for item in deck:
    List.append(item[0])

cabin=DataFrame(List,columns=['Cabin'])
sns.catplot('Cabin',data=cabin,kind='count',palette='winter_d')#pallete explicitly mentions what color should your plot be.

#if you dont want any particular cabin from the data you can redefine the data
cabin1=cabin[cabin.Cabin!='T']
sns.catplot('Cabin',data=cabin1,kind='count',palette='summer_d') #this elemenates 'T' cabin from plot

'''3.) Where did the passengers come from?'''
#embarked is citites where passengers got on
sns.catplot('Embarked',data=titanic,kind='count',hue='Pclass',order=['C','S','Q'])#to analyse passengers class from different boarding locations


'''Part3'''
'''4.) Who was alone and who was with family?'''
titanic.head()
#in data sibsp column indicates if passengers wer travelling with siblings are not, Parch column indicates if the passengers
#where travelling with parents or child
titanic['Alone']=titanic.SibSp + titanic.Parch
# if Alone column has anything but 0 means they had some one from the family travelling with them
titanic['Alone'].loc[titanic['Alone']>0]='With family'
titanic['Alone'].loc[titanic['Alone']==0]='With out family' #dont worry about the error it is the systems indication
#that you are working on a copy of data and not the original
titanic.head()
sns.catplot('Alone',data=titanic,kind='count',palette='Blues_d') #palette _d to show darker colors
plt.show()

'''Conclusion'''
#Who survived
titanic['Survivor']=titanic.Survived.map({0:'No',1:'Yes'})
titanic.head()
sns.catplot('Survivor',data=titanic,kind='count')
plt.show()
#WHAT FACTORS HELPED SURVIVAL OF THE PASSENGERS DURING THE SINKING OF TITANIC
#is class the reason
sns.catplot('Survivor',data=titanic,hue='Pclass',kind='count')
plt.show()
'''from the graph we can say that among thos who did not survive the count was high for 3rd class but among those 
who survived the count is not so different based on class it could be because the count of passengers traveling was
among third class or caz women and children are let first because 3rd class has more male'''
sns.catplot('Survivor',data=titanic,hue='person',kind='count')
plt.show()
'''Only a few male survived'''
sns.catplot('Pclass',data=titanic,kind='count')
plt.show()
sns.catplot('Pclass',data=titanic,hue='person',kind='count')
plt.show()
'''And yes 3rd class had more passengers compared to other classes and among them male where high'''
'''In short '''
sns.catplot('Pclass','Survived',data=titanic,hue='person',kind='bar')
plt.show()

'''These plots give a good idea but we can do a linear regression between age and survival to understand if age
has any influence too'''

sns.lmplot('Age','Survived',data=titanic)
plt.show()
#indicates younger the passenger greater are the chances of survival, we can analyse in diff classes as
sns.lmplot('Age','Survived',hue='Pclass',data=titanic,palette='winter_d')
plt.show()
# we can bin the data during regression based on generations
generations=[10,20,30,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic,palette='winter_d',x_bins=generations)
plt.show()#show high deviation in older ppl
#with respect to gender and age
generations=[10,20,30,40,60,80]
sns.lmplot('Age','Survived',hue='Sex',data=titanic,palette='winter_d',x_bins=generations)
plt.show()#show older female have more survival rate than older male, among female older had more chance
#at survival than younger

'''Did traveling alone have any effect on survival'''

sns.catplot('Survived',data=titanic,hue='Alone',kind='count')
plt.show()#with family survived more, did class have any effect on it
sns.catplot('Pclass','Survived',data=titanic,hue='Alone',kind='bar')
plt.show()#yes 1st class survived more what about gender and age impact
sns.catplot('person','Survived',data=titanic,hue='Alone',kind='bar')
plt.show()#female passengers travelling with family survived more

'''Did traveling on a particular deck have any effect'''
deck1=titanic['Cabin']
List1=[]
for item in deck1:
    if pd.isnull(item):
        List1.append('UN')
    else:
        List1.append(item[0])
Deck=DataFrame(List1)
titanic['Deck']=Deck
sns.catplot('Deck','Survived',data=titanic,hue='Pclass',kind='bar')
plt.show()#shows that 3rd class passengers in deck E were able to survive the most
sns.catplot('Deck','Survived',data=titanic,hue='person',kind='bar')
plt.show()#is inconclusive
sns.catplot('Deck','Survived',hue='Sex',data=titanic,kind='bar')
plt.show()
