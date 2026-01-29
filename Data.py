import numpy as np
import pandas as pd
import os

# Cria pasta para salvar os arquivos se não existir
if not os.path.exists('bases_de_dados'):
    os.makedirs('bases_de_dados')

print("Gerando bases de dados das aplicações...\n")

# ==============================================================================
# 1. DADOS DA EQUAÇÃO INTEGRAL DE FREDHOLM
# ==============================================================================
print("1. Gerando dados de Fredholm...")

def gerar_dados_fredholm():
    # Parâmetros
    n_points = 100
    x = np.linspace(0, 1, n_points)
    lambda_val = 0.1
    
    # Funções do problema: u(x) = sin(x) + λ * ∫ x*y*u(y) dy
    f_func = np.sin(x)
    
    # Inicialização (u0 = f)
    u = f_func.copy()
    
    # Processo iterativo (simplificado para geração de dados)
    # Executamos iterações até convergir
    for _ in range(50):
        u_novo = np.zeros_like(x)
        for i, xi in enumerate(x):
            # K(x,y,u) = xi * y * u
            # Integral via trapézio
            integrando = xi * x * u
            integral = np.trapz(integrando, x)
            u_novo[i] = f_func[i] + lambda_val * integral
        u = u_novo.copy()
    
    # Criação do DataFrame
    df_fredholm = pd.DataFrame({
        'x': x,
        'f_x_fonte': f_func,       # Termo fonte f(x)
        'u_x_solucao': u,          # Solução encontrada u(x)
        'erro_teorico': np.abs(u - (np.sin(x) + (0.1*x)/(1-0.1/3))) # Apenas ilustrativo
    })
    
    # Salva em CSV
    caminho = 'bases_de_dados/fredholm_data.csv'
    df_fredholm.to_csv(caminho, index=False)
    print(f"   -> Salvo em: {caminho}")
    return df_fredholm

# ==============================================================================
# 2. DADOS DO TEOREMA DE PICARD-LINDELÖF (EDO)
# ==============================================================================
print("2. Gerando dados de Picard-Lindelöf...")

def gerar_dados_picard():
    # Problema: dx/dt = x, x(0) = 1 (Solução exata: e^t)
    t0, x0 = 0, 1
    t = np.linspace(t0, 1, 100)
    
    # Armazenar histórico das iterações para o CSV
    iteracoes_dict = {'t': t, 'x_exato': np.exp(t)}
    
    # Iteração 0
    x_atual = np.ones_like(t) * x0
    iteracoes_dict['iteracao_0'] = x_atual.copy()
    
    # 5 iterações de Picard
    for n in range(1, 6):
        x_novo = np.zeros_like(t)
        for i, ti in enumerate(t):
            # Integração de 0 a ti
            if i == 0:
                x_novo[i] = x0
                continue
                
            t_slice = t[:i+1]
            x_slice = x_atual[:i+1]
            
            # f(t,x) = x neste caso
            integral = np.trapz(x_slice, t_slice)
            x_novo[i] = x0 + integral
            
        x_atual = x_novo.copy()
        iteracoes_dict[f'iteracao_{n}'] = x_atual
        
    # Criação do DataFrame
    df_picard = pd.DataFrame(iteracoes_dict)
    
    # Salva em CSV
    caminho = 'bases_de_dados/picard_data.csv'
    df_picard.to_csv(caminho, index=False)
    print(f"   -> Salvo em: {caminho}")
    return df_picard

# ==============================================================================
# 3. DADOS DO MÉTODO DE NEWTON
# ==============================================================================
print("3. Gerando dados do Método de Newton...")

def gerar_dados_newton():
    # Problema: f(x) = x^2 - 2 (Raiz de 2)
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    
    x_atual = 1.5 # Chute inicial
    dados_iteracoes = []
    
    # Executa iterações
    for n in range(10):
        fx = f(x_atual)
        dfx = df(x_atual)
        
        # Armazena dados atuais
        dados_iteracoes.append({
            'iteracao': n,
            'x_n': x_atual,
            'f_x': fx,
            'f_linha_x': dfx,
            'erro_estimado': abs(x_atual - np.sqrt(2))
        })
        
        # Passo de Newton
        if abs(dfx) < 1e-10: break
        x_novo = x_atual - fx/dfx
        
        if abs(x_novo - x_atual) < 1e-10:
            x_atual = x_novo
            # Adiciona a última iteração (convergida)
            dados_iteracoes.append({
                'iteracao': n+1, 'x_n': x_atual, 'f_x': f(x_atual), 
                'f_linha_x': df(x_atual), 'erro_estimado': abs(x_atual - np.sqrt(2))
            })
            break
            
        x_atual = x_novo
        
    # Criação do DataFrame
    df_newton = pd.DataFrame(dados_iteracoes)
    
    # Salva em CSV
    caminho = 'bases_de_dados/newton_data.csv'
    df_newton.to_csv(caminho, index=False)
    print(f"   -> Salvo em: {caminho}")
    return df_newton

# Executa as funções
if __name__ == "__main__":
    df1 = gerar_dados_fredholm()
    df2 = gerar_dados_picard()
    df3 = gerar_dados_newton()
    
    print("\nRESUMO DAS BASES GERADAS:")
    print("-" * 50)
    print("1. Fredholm (amostra):")
    print(df1.head(3))
    print("\n2. Picard (amostra):")
    print(df2[['t', 'iteracao_0', 'iteracao_5', 'x_exato']].head(3))
    print("\n3. Newton (completo):")
    print(df3)
    print("-" * 50)
    print(f"Arquivos disponíveis na pasta '{os.path.abspath('bases_de_dados')}'")
