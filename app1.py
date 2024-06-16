import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
dataset = pd.read_csv('RBC .csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree = 6)
X_poly = polynomial_reg.fit_transform(X)

linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)
age=None
st.header("Polynomial Regression to RBC Count")
try:
    age=float(st.text_input("Enter the age"))
except ValueError:
    st.error("Please enter valid age")
if age is not None:
    linear_reg.predict([[age]])
    p=linear_reg_2.predict(polynomial_reg.fit_transform([[age]]))
    if st.button("Calculate"):
        st.success("Successfully predict!")
        st.subheader("Expected RBC count")
        st.write(p[0])