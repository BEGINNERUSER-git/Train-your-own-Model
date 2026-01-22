import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
class Visualization:
    def __init__(self,df,X,y):
        self.df=df
        self.X=X
        self.y=y

    def scatter_plot(self,title="Scatter plot"):
        for col in self.X:
                 
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=self.df,x=self.df[col],y=self.df[self.y],hue=self.df[self.y],palette="viridis")
            plt.title(title)
            plt.xlabel("Feature 1: " + col)
            plt.ylabel("Feature 2: " + self.y)
            st.pyplot(plt)

    def histogram(self,title="Histogram"):
        for col in self.X:
            plt.figure(figsize=(10,6))
            sns.histplot(self.df[col],bins=30,kde=True,color='blue')
            plt.title(title)
            plt.xlabel("Values: " + col)
            plt.ylabel("Frequency")
            st.pyplot(plt)
    def box_plot(self,title="Box plot"):
        for col in self.X:
            plt.figure(figsize=(8,6))
            sns.boxplot(y=self.df[col],color='orange')
            plt.title(title)
            plt.ylabel("Values: " + col)
            st.pyplot(plt)
    def heatmap(self,title="Heatmap"):
            plt.figure(figsize=(10,8))
            sns.heatmap(self.df.corr(),annot=True,cmap='coolwarm',fmt=".2f")
            plt.title(title)
            st.pyplot(plt)
    
