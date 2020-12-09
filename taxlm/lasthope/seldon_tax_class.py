class linear_model:
    """ Перед использованием класса убедитесь, что отобраны адекватные объясняемые переменные.
    Рост количества проверок ведет к росту ошибок 1 рода
    Принимает на вход таблицу пандас и название целевой переменной
    data - вносим данные
    real_data - вносим реальные данные, если в data внесли относительные
    target_var - целевая переменная """

    def __init__(self, data, target_var, real_data='pep', drop_outl=True):
        if drop_outl:
            data[target_var] = data[target_var][
                (data[target_var] <= (data[target_var].mean() + 2 * data[target_var].std())) &
                (data[target_var] >= (data[target_var].mean() - 2 * data[target_var].std()))]

            print("Удалено {0} наблюдений из {1}".format(data[target_var].shape[0] - data[target_var].dropna().shape[0],
                                                         data[target_var].shape[0]))
            data.dropna(inplace=True)

        self.real_data = real_data
        self.data = data
        self.target_var = target_var

    def simple_model(self, quartal=False):
        if quartal:
            pred = self.real_data.resample('QS', axis=0).sum()[[self.target_var]].copy()

            pred[self.target_var + '_return'] = pred[self.target_var] / pred[self.target_var].shift(4)
            pred[self.target_var + '_shift'] = pred[self.target_var].shift(3)

            pred[self.target_var + '_predict'] = pred[self.target_var + '_return'] * pred[self.target_var + '_shift']
            pred[self.target_var + '_predict'] = pred[self.target_var + '_predict'].shift(1)
            pred.dropna(inplace=True)

        else:
            pred = self.real_data.loc[:, [self.target_var]].copy()
            pred[self.target_var + '_return'] = pred[self.target_var] / pred[self.target_var].shift(12)
            pred[self.target_var + '_shift'] = pred[self.target_var].shift(11)
            pred[self.target_var + '_predict'] = pred[self.target_var + '_return'] * pred[self.target_var + '_shift']
            pred[self.target_var + '_predict'] = pred[self.target_var + '_predict'].shift(1)
            pred.dropna(inplace=True)

        self.func_graph_pr(model_predict=pred[self.target_var + '_predict'].values, index=pred.index,
                           data_base=pred[self.target_var], model_name='Примитивная модель', unit=' млрд, руб')

    def correlation_table(self, size_x=15, size_y=7, corr_method='pearson', min_corr=0.15):
        """ Рисует таблицу корреляций на полученных данных
        size_x - размер оси X
        size_y - размер оси Y
        corr_method - Метод построение таблицы корреляции (дефолт = Пирсон) """
        import seaborn as sns
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        # Ставим целевую переменную в первую колонку
        list_cols = self.data.columns.tolist()
        index = list_cols.index(self.target_var)
        list_col = [list_cols[index]] + list_cols[:index] + list_cols[index + 1:]
        self.data = self.data.loc[:, list_col]

        # выбираем индексы с хорошей корреляцией
        indexes = self.data.corr().iloc[1:, 0][
            (self.data.corr().iloc[1:, 0] > min_corr) | (self.data.corr().iloc[1:, 0] < -min_corr)].index
        indexes = np.append([self.target_var], indexes)
        corr_table = self.data[indexes]
        corr_table.dropna(inplace=True)

        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('figure', titlesize=10)

        fig = plt.figure(figsize=(size_x, size_y))
        sns.heatmap(corr_table.corr(method=corr_method), vmin=-1, vmax=1, linewidths=0.5, cmap='vlag', annot=True)
        plt.yticks(rotation=0)
        plt.show()

    def sequences_with_max_corr(self, min_corr=.3, max_col=.7, min_len_seq=3, max_len_seq=5):
        """функция формирует последовательности признаков для подачи в регрессию, используя таблицу корреляций
        min_corr - минимальная корреляция экзогенных переменных с эндогенной,
        max_col - максимально допустимая коллинеарность признаков
        min_len_seq - минимальная длина последовательности признаков
        max_len_seq - максимальная длина последовательности признаков"""
        import statsmodels.formula.api as smf
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        import seaborn as sns
        import numpy as np
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        # Ставим целевую переменную в первую колонку
        list_cols = self.data.columns.tolist()
        index = list_cols.index(self.target_var)
        list_col = [list_cols[index]] + list_cols[:index] + list_cols[index + 1:]
        self.data = self.data.loc[:, list_col]

        # выбираем индексы с хорошей корреляцией
        indexes = self.data.corr().iloc[1:, 0][
            (self.data.corr().iloc[1:, 0] > min_corr) | (self.data.corr().iloc[1:, 0] < -min_corr)].index

        # формируем все возможные неколлинеарные последовательности признаков
        corr_table = self.data[indexes].corr()[
            ((self.data[indexes].corr() < max_col) & (self.data[indexes].corr() > -max_col)) | (
                    self.data[indexes].corr() == 1.)]
        combs = []
        for i in range(min_len_seq, max_len_seq + 1):
            for comb in itertools.combinations(corr_table.index.values, i):
                if corr_table.loc[comb, comb].isnull().any().any() == False:
                    comb = np.append(self.data.filter(regex=self.target_var).columns[0], np.array(comb, dtype=object))
                    combs.append(comb)  # заменил на нампай аппенд
        return combs

    def sequences_selection(self, p_value_max=0.05, R_2_min=.3, min_corr_self=0.3, max_col_self=0.7, min_len_seq=3,
                            max_len_seq=4):
        ''' Строим регрессии на неколлинеарных последовательностях признаков
        и отбираем тем, что удовлетворяют условиям
        p_value_max - минимальное значение p_value
        R_2_min - минимальный R^2
        Возвращает последовательность признаков с максимальным R^2
        А также принтит R^2 для всех вошедших последовательностей'''
        import statsmodels.formula.api as smf
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        import seaborn as sns
        import numpy as np
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        # Ставим целевую переменную в первую колонку
        list_cols = self.data.columns.tolist()
        index = list_cols.index(self.target_var)
        list_col = [list_cols[index]] + list_cols[:index] + list_cols[index + 1:]
        self.data = self.data.loc[:, list_col]

        # подгружаем список потенциальных комбинаций
        print('Отбираем последователи признаков', '\n')
        combs = self.sequences_with_max_corr(min_corr=min_corr_self, max_col=max_col_self, min_len_seq=min_len_seq,
                                             max_len_seq=max_len_seq)
        self.list_exog = []
        self.list_r2 = []
        max_r_2 = 0
        self.opt_seq = []
        print('Строим регрессии, из них подходят условию:', '\n')
        for element in combs:
            mod = sm.OLS(self.data[element[0]], sm.add_constant(self.data[list(element[1:])], prepend=False))
            res = mod.fit(cov_type='HC1')
            bool_var_p = np.isin(False, np.array(res.pvalues[:-1] < p_value_max))
            bool_var_r = np.isin(False, np.array(res.rsquared > R_2_min))
            if (bool_var_p == False) & (bool_var_r == False):
                self.list_exog = np.append(self.list_exog, list(element[1:]))
                self.list_r2 = np.append(self.list_r2, res.rsquared)
                print('Список предикторов: {0}, R^2: {1}'.format(list(element[1:]), res.rsquared))
                if res.rsquared > max_r_2:
                    max_r_2 = res.rsquared
                    self.opt_seq = list(element[1:])

        return self.opt_seq

    def regression_summary(self, exog_var_list='best_choice', size_x=15, size_y=7, p_value_max=0.05, R_2_min=.3,
                           min_corr_self=0.3, max_col_self=0.7):
        import statsmodels.formula.api as smf
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        import seaborn as sns
        import numpy as np
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import pylab
        import scipy as sc

        ''' Строит необходимые графики и таблицы для понимания регресси по заданным параметрам
        exog_var_list - Лист наименований регрессоров (str)
        size_x - размер оси X
        size_y - размер оси Y '''

        if exog_var_list == 'best_choice':
            exog_var_list = self.sequences_selection(p_value_max=p_value_max, R_2_min=R_2_min,
                                                     min_corr_self=min_corr_self, max_col_self=max_col_self)

        sns.set(style='whitegrid')
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # font size of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('figure', titlesize=10)

        print('\n', '\n', 'Рисуем распределение данных', '\n', '\n')
        fig = plt.figure(figsize=(size_x, size_y))
        sns.pairplot(self.data.loc[:, exog_var_list])
        plt.show()

        print('\n', '\n', 'Рисуем гистограмму распределения целевой переменной', '\n', '\n')
        plt.figure(figsize=(15, 7))
        self.data[self.target_var].plot.hist()
        plt.xlabel(self.target_var, fontsize=14)
        plt.show()

        print('\n', '\n', 'Регрессия', '\n', '\n')
        mod = sm.OLS(self.data[self.target_var], sm.add_constant(self.data[exog_var_list], prepend=False))
        fitted = mod.fit(cov_type='HC1')
        print(fitted.summary(), '\n')

        print('\n', '\n', 'Распределение остатков', '\n', '\n')
        plt.figure(figsize=(16, 7))
        plt.subplot(121)
        sc.stats.probplot(fitted.resid, dist="norm", plot=pylab)
        plt.subplot(122)
        np.log(fitted.resid).plot.hist()
        plt.xlabel('Residuals', fontsize=14)
        plt.show()
        print('\n', '\n')
        print(
            'Breusch-Pagan test на гомоскедастичность: p=%f' % sms.het_breuschpagan(fitted.resid, fitted.model.exog)[1])

    def func_graph_pr(self, model_predict, index, data_base='kurva', model_name='Линейная модель', unit=' млрд, руб'):
        """ Вспомогательный метод, рисует график качества на основе входных данных
        model_predict - лист ответов регрессии
        data_base_1 - лист реальных ответов
        model_name - наименование модели """
        import statsmodels.formula.api as smf
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        import seaborn as sns
        import numpy as np
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error

        sns.set(style='whitegrid')
        fig = plt.figure(figsize=(20, 10))
        plt.rc('font', size=20)  # controls default text sizes
        plt.rc('axes', titlesize=20)  # fontsize of the axes title
        plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
        plt.rc('legend', fontsize=19)  # legend fontsize
        plt.rc('figure', titlesize=26)
        plt.xlabel('Ось времени')
        plt.ylabel(self.target_var + unit)

        plt.plot(index, data_base, label="Факт")
        plt.plot(index, model_predict, "r", label="Прогноз")
        plt.fill_between(index, data_base, model_predict, facecolor='b', alpha=0.3)
        plt.legend(loc="best")

        plt.title("{0}\n Средняя абсолютная ошибка {1} \n Средняя относительная ошибка {2:0.3}".format(model_name,
                                                                                                       round(
                                                                                                           mean_absolute_error(
                                                                                                               model_predict,
                                                                                                               data_base),
                                                                                                           3),
                                                                                                       mean_absolute_error(
                                                                                                           model_predict,
                                                                                                           data_base) / np.mean(
                                                                                                           model_predict) * 100,
                                                                                                       unit))

    def linear_model_results(self, exog_var_list='best_choice', p_value_max=0.05, R_2_min=0.3, num_points=4,
                             unit=', млрд. руб', data_split=False, relative_data=False, period=1, quarter=True,
                             min_corr_self=.3, max_col_self=.7):
        ''' Рисует график качества модели
        exog_var_list - список переменных, если best_choice, то отбирает автоматически
        p_value_max - максимум пи вэлью для отбора признаков
        R_2_min - минимальное значение Р_2 для отбора признаков
        num_points - число точек для теста
        unit - единицы измерения данных
        data_split - Нужно делить данные на тест и трейн: True/False
        relative_data - Данные представлены в относительных величинах: True/False
        period - период расчета относительных величин (нужно фиксить)
        quarter - переводить данные в кварталы True/False'''
        import statsmodels.formula.api as smf
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        import seaborn as sns
        import numpy as np
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        if data_split:

            # Подгружаем последовательность
            if exog_var_list == 'best_choice':
                exog_var_list = self.sequences_selection(p_value_max=p_value_max, R_2_min=R_2_min,
                                                         min_corr_self=min_corr_self, max_col_self=max_col_self)

            # разбиваем весь датасет на тренировочную и тестовую выборку
            y_test = self.data.loc[self.data.index >= self.data.index[-num_points], [self.target_var]]
            y_train = self.data.loc[self.data.index < self.data.index[-num_points], [self.target_var]]

            x_train = self.data.loc[self.data.index < self.data.index[-num_points], exog_var_list]
            x_train = statsmodels.tools.tools.add_constant(x_train, prepend=False, has_constant='skip')
            x_test = self.data.loc[self.data.index >= self.data.index[-num_points], exog_var_list]
            x_test = statsmodels.tools.tools.add_constant(x_test, prepend=False, has_constant='skip')

            predictions_test = pd.DataFrame()
            predictions_train = pd.DataFrame()
            coef_table = pd.DataFrame(index=self.data.columns)

            # Обучаем линейную модель
            model = LinearRegression()
            model.fit(x_train, y_train)

            # Заносим предикты на тесте
            predict_test = model.predict(x_test)
            predictions_test['lr_pred'] = predict_test.T[0]
            predictions_test.index = y_test.index

            # Заносим предикты на трейн
            predict_train = model.predict(x_train)
            predictions_train['lr_pred'] = predict_train.T[0]
            predictions_train.index = y_train.index

            # Заносим коэф-ты на трейн
            self.coef_tab = pd.DataFrame(data = model.coef_[0], index = np.array(x_train.columns), columns = ['coef'])
            print(self.coef_tab)

            if relative_data:
                # Подгружаем датасет реальных данных
                # Переводим значения в кварталы, если нужно
                if quarter:
                    real = self.real_data.loc[:, [self.target_var]]
                    real = real.resample('QS', axis=0).sum()
                else:
                    real = self.real_data.loc[:, [self.target_var]]
                # Заводим вспомогательный датасет для графика train
                real = real.loc[:, [self.target_var]]
                real.rename(columns={self.target_var: self.target_var + 'real'}, inplace=True)
                aux_train = pd.DataFrame(index=y_train.index.to_timestamp(), data=predict_train, columns=[self.target_var], )
                aux_train.index = aux_train.index - pd.DateOffset(years=period)
                aux_train = pd.merge(aux_train, real, how='left', left_index=True, right_index=True)
                aux_train.index = aux_train.index + pd.DateOffset(years=period)
                aux_train['real_predict'] = aux_train[self.target_var] * aux_train[self.target_var + 'real']
                aux_train.drop(columns=[self.target_var + 'real'], inplace=True)
                aux_train = pd.merge(aux_train, real, how='left', left_index=True, right_index=True)

                # Заводим вспомогательный датасет для графика test
                real = self.real_data.loc[:, [self.target_var]]
                real.rename(columns={self.target_var: self.target_var + 'real'}, inplace=True)
                aux_test = pd.DataFrame(index=y_test.index.to_timestamp(), data=predict_test, columns=[self.target_var])
                aux_test.index = aux_test.index - pd.DateOffset(years=period)
                aux_test = pd.merge(aux_test, real, how='left', left_index=True, right_index=True)
                aux_test.index = aux_test.index + pd.DateOffset(years=period)
                aux_test['real_predict'] = aux_test[self.target_var] * aux_test[self.target_var + 'real']
                aux_test.drop(columns=[self.target_var + 'real'], inplace=True)
                aux_test = pd.merge(aux_test, real, how='left', left_index=True, right_index=True)

                # Рисуем график качества
                self.func_graph_pr(model_predict=aux_train['real_predict'].values,
                                   data_base=aux_train[[self.target_var + 'real']].values.T[0], index=aux_train.index,
                                   unit=unit, model_name='Линейная модель, трейн')
                self.func_graph_pr(model_predict=aux_test['real_predict'].values,
                                   data_base=aux_test[[self.target_var + 'real']].values.T[0], index=aux_test.index,
                                   unit=unit, model_name='Линейная модель, тест')
                return aux_train, aux_test

            else:
                # Рисуем график качества

                self.func_graph_pr(model_predict=predictions_train.values.T[0], data_base=y_train.values.T[0],
                                   index=predictions_train.index, unit=unit, model_name='Линейная модель, трейн')
                self.func_graph_pr(model_predict=predictions_test.values.T[0], data_base=y_test.values.T[0],
                                   index=predictions_test.index, unit=unit, model_name='Линейная модель, тест')
                return predictions_train, predictions_test

        else:

            # Подгружаем последовательность
            if exog_var_list == 'best_choice':
                exog_var_list = self.sequences_selection(p_value_max=p_value_max, R_2_min=R_2_min)

            # Обучаем регрессию на подгруженной последовательности
            mod = sm.OLS(self.data[self.target_var], sm.add_constant(self.data[exog_var_list], prepend=False))
            fitted = mod.fit(cov_type='HC1')

            # Определяем коэффициенты и предсказания модели
            self.reg_coef = pd.DataFrame(fitted.params, columns=['reg_coef'])
            self.fittedvalues = pd.DataFrame(fitted.fittedvalues, columns=['fittedvalues'])

            print(self.reg_coef)
            # Рисуем график качества
            self.func_graph_pr(model_predict=self.fittedvalues.values.T[0],
                               data_base=self.data[[self.target_var]].values.T[0],
                               index=self.fittedvalues.index.to_timestamp(), unit=unit,
                               model_name='Линейная модель')

            if relative_data:
                # Подгружаем датасет реальных данных
                # Переводим значения в кварталы, если нужно
                if quarter:
                    real = self.real_data.loc[:, [self.target_var]]
                    real = real.resample('QS', axis=0).sum()
                    # real.index = real.index.to_period('Q')

                else:
                    real = self.real_data.loc[:, [self.target_var]]

                # Заводим вспомогательный датасет для графика train
                real = real.loc[:, [self.target_var]]
                real.rename(columns={self.target_var: self.target_var + 'real'}, inplace=True)

                aux = pd.DataFrame(index=self.fittedvalues.index.to_timestamp(), data=self.fittedvalues.values,
                                   columns=[self.target_var])
                aux.index = aux.index - pd.DateOffset(years=period)
                aux = pd.merge(aux, real, how='left', left_index=True, right_index=True)
                aux.index = aux.index + pd.DateOffset(years=period)
                aux['real_predict'] = aux[self.target_var] * aux[self.target_var + 'real']
                aux.drop(columns=[self.target_var + 'real'], inplace=True)
                aux = pd.merge(aux, real, how='left', left_index=True, right_index=True)

                # Рисуем график качества
                self.func_graph_pr(model_predict=aux['real_predict'].values,
                                   data_base=aux[[self.target_var + 'real']].values.T[0], index=aux.index, unit=unit,
                                   model_name='Линейная модель, трейн')
                return aux
