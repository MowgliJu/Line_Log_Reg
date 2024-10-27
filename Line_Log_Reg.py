import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.metrics import mean_squared_error

# выбор регрессии
st.title('Линейная и Логистическая регрессии')
st.write('Выберите регрессию')

regression_type = st.sidebar.radio("Выберите тип регрессии:", ("Линейная регрессия", "Логистическая регрессия"))

# загрузка CSV файла
uploader_train_file = st.sidebar.file_uploader("Нажми сюда для загрузки файла с обучающими данными", type="csv")
if uploader_train_file is not None:
    df_train = pd.read_csv(uploader_train_file)
    st.write("Обучающий набор данных:")
    st.write(df_train.head())
else:
    st.write("Пожалуйста, загрузите файл с обучающими данными.")

# Загрузка файла с тестовыми данными
uploader_test_file = st.sidebar.file_uploader("Нажми сюда для загрузки файла с тестовыми данными", type="csv")
if uploader_test_file is not None:
    df_test = pd.read_csv(uploader_test_file)
    st.write("Тестовый набор данных:")
    st.write(df_test.head()) 
else:
    st.write("Пожалуйста, загрузите файл с тестовыми данными.")

# нормализация данных
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

epochs = st.number_input(
    "Введите количество эпох",
    min_value=0,
    max_value=1000,
    format="%d",
    help="Введите целое число от 0 до 1000.",
)

learning_rate = st.number_input(
    "Введите скорость обучения",
    min_value=0.0,
    max_value=10.0,
    format="%f",
    help="Введите целое число от 0 до 10.",
)

# Линейная регрессия
if regression_type == "Линейная регрессия":

    class LinReg:
        def __init__(self, learning_rate, epochs, feature_names):
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.feature_names = feature_names
            self.start_coef_ = np.random.uniform(-1, 1, size=len(feature_names))
            self.start_intercept_ = np.random.uniform(-1, 1)
            self.coef_ = self.start_coef_.copy()
            self.intercept_ = self.start_intercept_

        def fit(self, X, y):
            X = np.array(X)
            y = np.array(y)

            self.display_w(self.start_coef_, self.start_intercept_, "Стартовые веса")

            for epoch in range(self.epochs):
                self.coef_ -= self.learning_rate * self.grad(X, y).mean(axis=1)
                self.intercept_ -= self.learning_rate * self.derivative_w0(X, y).mean()

            self.display_w(self.coef_, self.intercept_, "Обновленные веса")

        def display_w(self, coef, intercept, title):
            coef_dict = {self.feature_names[i]: coef[i] for i in range(len(coef))}
            st.write(f'{title}: {coef_dict}, Интерсепт: {intercept}')

        def grad(self, X, y):
            return np.array([self.derivative_w1(X, y), self.derivative_w2(X, y)])
        
        def predict(self, X):
            return X @ self.coef_ + self.intercept_

        def derivative_w0(self, X, y):
            return -2 * (y - self.predict(X))

        def derivative_w1(self, X, y):
            return -2 * X[:, 0] * (y - self.predict(X))

        def derivative_w2(self, X, y):
            return -2 * X[:, 1] * (y - self.predict(X))

# После того, как обе модели обучены:
if regression_type == "Линейная регрессия":
    
    model = LinReg(learning_rate=0.01, epochs=1000, feature_names=X_train.columns.tolist())
    model.fit(X_train.values, y_train.values)

    # Обучаем модель из sklearn
    sklearn_model = SklearnLinReg()
    sklearn_model.fit(X_train, y_train)

    # Предсказания
    y_pred_custom = model.predict(X_test.values)
    y_pred_sklearn = sklearn_model.predict(X_test)

    # Оценка качества
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    st.write("Среднеквадратичная ошибка (MSE) вашей линейной регрессии:", mse_custom)
    st.write("Среднеквадратичная ошибка (MSE) линейной регрессии из sklearn:", mse_sklearn)



# Логистическая регрессия
elif regression_type == "Логистическая регрессия":
    class LogReg:
        def __init__(self, learning_rate, epochs, feature_names):
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.feature_names = feature_names
            self.start_coef_ = np.random.uniform(-1, 1, size=len(feature_names))
            self.start_intercept_ = np.random.uniform(-1, 1)
            self.coef_ = self.start_coef_.copy()
            self.intercept_ = self.start_intercept_
            

        def fit(self, X, y):
            X = np.array(X)
            y = np.array(y)

            # стартовые веса
            self.display_w(self.start_coef_, self.start_intercept_, "Стартовые веса")

            for epoch in range(self.epochs):
                self.coef_ -= self.learning_rate * self.grad(X, y).mean(axis=1)
                self.intercept_ -= self.learning_rate * self.derivative_w0(X, y).mean()

            # обновленные веса
            self.display_w(self.coef_, self.intercept_, "Обновленные веса")

            x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
            y_boundary = -self.coef_[0] / self.coef_[1] * x - self.intercept_ / self.coef_[1]

            fig, ax = plt.subplots()
            ax.clear()
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=4)
            ax.plot(x, y_boundary, color='red', label='Граница решения')
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            ax.legend()
            st.pyplot(fig)

        def display_w(self, coef, intercept, title):
            # нужно вывести названия столбцов
            coef_dict = {self.feature_names[i]: coef[i] for i in range(len(coef))}
            st.write(f'{title}: {coef_dict}, Интерсепт: {intercept}')

        def grad(self, X, y):
            return np.array([self.derivative_w1(X, y), self.derivative_w2(X, y)])

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def predict_proba(self, X):
            return self.sigmoid(X @ self.coef_ + self.intercept_)

        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)

        def derivative_w0(self, X, y):
            return -(y - self.predict_proba(X))

        def derivative_w1(self, X, y):
            predictions = self.predict_proba(X)
            return -X[:, 0] * (y - predictions)

        def derivative_w2(self, X, y):
            predictions = self.predict_proba(X)
            return -X[:, 1] * (y - predictions) 
        
if regression_type == "Логистическая регрессия":
    # Обучаем вашу модель
    log_reg = LogReg(learning_rate=0.01, epochs=1000, feature_names=X_train.columns.tolist())
    log_reg.fit(X_train.values, y_train.values)

    # Обучаем модель из sklearn
    sklearn_model = SklearnLogReg()
    sklearn_model.fit(X_train, y_train)

    # Предсказания
    y_pred_custom = log_reg.predict(X_test.values)
    y_pred_sklearn = sklearn_model.predict(X_test)

    # Оценка точности
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

    st.write("Точность вашей логистической регрессии:", accuracy_custom)
    st.write("Точность логистической регрессии из sklearn:", accuracy_sklearn)
        

# Инициализация и обучение модели
if regression_type == "Линейная регрессия":
    model = LinReg(learning_rate, epochs, X_train.columns)
    model.fit(X_train, y_train)

elif regression_type == "Логистическая регрессия":
    model = LogReg(learning_rate, epochs, X_train.columns)
    model.fit(X_train, y_train)
