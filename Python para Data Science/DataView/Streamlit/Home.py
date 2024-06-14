import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(layout = 'wide')

#----------------------------------------------------------------
def formata_numero(valor, prefixo = ''):
    for unidade in ['', 'milhões']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}'
        valor /= 1000 # Divide por 1000
    return f'{prefixo} {valor:.2f} {unidade}'

#----------------------------------------------------------------

url = 'https://labdados.com/produtos'
#Criando um filtro direto na url da API com escolha do usuário
regioes = ['Brasil', 'Centro-Oeste','Nordeste', 'Sudeste', 'Sul']
st.sidebar.title('Filtros')
regiao = st.sidebar.selectbox('Região', regioes) #Variavel para armazenar a escolha do usuário

if regiao == 'Brasil':
    regiao = ''

todos_anos = st.sidebar.checkbox('Dados de todo o período', value = True)
if todos_anos:
    ano = ''
else:
    ano = st.sidebar.slider('Ano',2020,2023)

query_string = {'regiao':regiao.lower(), 'ano':ano}
response = requests.get(url,params = query_string)

# Transformando a requisição em json e depois convertendo para df
dados = pd.DataFrame.from_dict(response.json())

dados['Data da Compra'] = pd.to_datetime(dados['Data da Compra'], format = '%d/%m/%Y')

#Filtro dos vendedores
filtro_vendedores = st.sidebar.multiselect('Vendedores', dados['Vendedor'].unique())
if filtro_vendedores:
    dados = dados[dados['Vendedor'].isin(filtro_vendedores)]

#----------------------------------------------------------------
# Tabelas
## Tabelas de Receita
receita_estados = dados.groupby('Local da compra')[['Preço']].sum()
receita_estados = dados.drop_duplicates(subset = 'Local da compra')[['Local da compra', 'lat', 'lon']].merge(receita_estados, left_on = 'Local da compra', right_index = True).sort_values('Preço', ascending = False)

receita_mensal = dados.set_index('Data da Compra').groupby(pd.Grouper(freq = 'M'))['Preço'].sum().reset_index()
receita_mensal['Ano'] = receita_mensal['Data da Compra'].dt.year
receita_mensal['Mes'] = receita_mensal['Data da Compra'].dt.month_name()

receita_categorias = dados.groupby('Categoria do Produto')[['Preço']].sum().sort_values('Preço', ascending = False)

## Tabelas de Quantidade de Vendas
vendas_estado = dados.groupby('Local da compra')[['Produto']].count()
vendas_estado = dados.drop_duplicates(subset = 'Local da compra')[['Local da compra', 'lat', 'lon']].merge(vendas_estado, left_on = 'Local da compra', right_index = True)

vendas_mensal = dados.set_index('Data da Compra').groupby(pd.Grouper(freq = 'M'))['Produto'].count().reset_index()
vendas_mensal['Ano'] = vendas_mensal['Data da Compra'].dt.year
vendas_mensal['Mes'] = vendas_mensal['Data da Compra'].dt.month_name()

maiores_vendas_estados = vendas_estado.sort_values('Produto', ascending=False)

vendas_produto = dados.groupby('Produto')[['Preço']].count()
vendas_produto['Quantidade'] = vendas_produto['Preço']

##Tabelas Vendedores
vendedores = pd.DataFrame(dados.groupby('Vendedor')['Preço'].agg(['sum','count']))

#----------------------------------------------------------------
# Gráficos
fig_mapa = px.scatter_geo(receita_estados,
                          lat = 'lat',
                          lon = 'lon',
                          scope = 'south america',
                          size = 'Preço',
                          template = 'seaborn',
                          hover_name = 'Local da compra',
                          hover_data = {'lat': False,'lon': False},
                          title = 'Receita por estado')


fig_receita_mensal = px.line(receita_mensal,
                              x = 'Mes',
                              y= 'Preço',
                              markers=True,
                              range_y = (0, receita_mensal.max()),
                              color = 'Ano',
                              line_dash = 'Ano',
                              title = 'Receita Mensal')
fig_receita_mensal.update_layout(yaxis_title = 'Receita')


fig_receita_estados = px.bar(receita_estados.head(), 
                             x = 'Local da compra',
                             y = 'Preço',
                             text_auto = True,
                             title = 'Top estados (Receita)')
fig_receita_estados.update_layout(yaxis_title = 'Receita')


fig_receita_categorias = px.bar(receita_categorias,
                                text_auto = True,
                                title = 'Receita por Categoria')
fig_receita_categorias.update_layout(yaxis_title = 'Receita')


fig_mapa_vendas = px.scatter_geo(vendas_estado,
                          lat = 'lat',
                          lon = 'lon',
                          scope = 'south america',
                          size = 'Produto',
                          template = 'seaborn',
                          hover_name = 'Local da compra',
                          hover_data = {'lat': False,'lon': False},
                          title = 'Vendas por estado')

fig_vendas_mensal = px.line(vendas_mensal,
                              x = 'Mes',
                              y= 'Produto',
                              markers=True,
                              range_y = (0, vendas_mensal.max()),
                              color = 'Ano',
                              line_dash = 'Ano',
                              title = 'Vendas Mensal')
fig_vendas_mensal.update_layout(yaxis_title = 'Vendas')

fig_maiores_vendas_mensal = px.bar(maiores_vendas_estados.head(),
                                   x = 'Local da compra',
                                   y = 'Produto',
                                   text_auto = True,
                                   title = 'Top estados (Vendas)')

fig_produtos = px.bar(vendas_produto,
                      x = vendas_produto.index,
                      y = 'Quantidade',
                      text_auto = True,
                      title = 'Quantidade de vendas por produto')

#----------------------------------------------------------------
st.title('DASHBOARD DE VENDAS')

aba0, aba1, aba2, aba3 = st.tabs(['Dados','Receita', 'Quantidade de vendas', 'Vendedores'])

with aba0: # Dados
    linhas = st.number_input('Número de Linhas',1,None,10)
    st.dataframe(dados.head(linhas))
    st.dataframe(vendas_produto)

with aba1: # Receita
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(),'R$'))
        st.plotly_chart(fig_mapa, use_container_width= True)
        st.plotly_chart(fig_receita_estados, use_container_width = True)

    with col2:
        st.metric('Quantidade de Vendas', dados.shape[0])
        st.plotly_chart(fig_receita_mensal, use_container_width= True)
        st.plotly_chart(fig_receita_categorias, use_container_width= True)

with aba2: # Quantidade de Vendas
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(),'R$'))
        st.plotly_chart(fig_mapa_vendas, use_container_width= True)
        st.plotly_chart(fig_maiores_vendas_mensal, use_container_width= True)

    with col2:
        st.metric('Quantidade de Vendas', dados.shape[0])
        st.plotly_chart(fig_vendas_mensal, use_container_width= True)
        st.plotly_chart(fig_produtos, use_container_width=True)

with aba3: #Vendedores
    qtd_vendedores = st.number_input('Quantidade de Vendedores', 2,10,5)
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(),'R$'))

        fig_receita_vendedores = px.bar(vendedores[['sum']].sort_values('sum',ascending=False).head(qtd_vendedores),
                                        x = 'sum',
                                        y = vendedores[['sum']].sort_values('sum',ascending=False).head(qtd_vendedores).index,
                                        text_auto = True,
                                        title = f'Top {qtd_vendedores} vendedores (receita)')
        st.plotly_chart(fig_receita_vendedores)

    with col2:
        st.metric('Quantidade de Vendas', dados.shape[0])
        fig_vendas_vendedores = px.bar(vendedores[['count']].sort_values('count',ascending=False).head(qtd_vendedores),
                                                x = 'count',
                                                y = vendedores[['count']].sort_values('count',ascending=False).head(qtd_vendedores).index,
                                                text_auto = True,
                                                title = f'Top {qtd_vendedores} vendedores (quantidade de vendas)')
        st.plotly_chart(fig_vendas_vendedores)


