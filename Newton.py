import numpy as np
import matplotlib.pyplot as plt

def metodo_newton_banach(f, df, ddf, x0, max_iter=20, tol=1e-10):
    """
    Método de Newton usando o Princípio da Contração de Banach.
    
    Encontra o zero de f(x) = 0 através da iteração:
    x_{n+1} = x_n - f(x_n)/f'(x_n)
    
    que é equivalente ao ponto fixo de F(x) = x - f(x)/f'(x)
    
    Parâmetros:
    - f: função f(x)
    - df: derivada f'(x)
    - ddf: segunda derivada f''(x)
    - x0: aproximação inicial
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência
    """
    
    iteracoes = [x0]
    erros = []
    residuos = []
    
    x_atual = x0
    
    print(f"{'Iter':<6} {'x_n':<15} {'f(x_n)':<15} {'Erro':<15}")
    print("-" * 55)
    print(f"{0:<6} {x_atual:<15.10f} {f(x_atual):<15.10e} {'---':<15}")
    
    for n in range(1, max_iter + 1):
        # Verifica se a derivada não é zero
        df_val = df(x_atual)
        if abs(df_val) < 1e-12:
            print("\nDerivada muito próxima de zero. Iteração parada.")
            break
        
        # Iteração de Newton: x_{n+1} = x_n - f(x_n)/f'(x_n)
        x_novo = x_atual - f(x_atual) / df_val
        
        # Calcula erro e resíduo
        erro = abs(x_novo - x_atual)
        residuo = abs(f(x_novo))
        
        erros.append(erro)
        residuos.append(residuo)
        iteracoes.append(x_novo)
        
        print(f"{n:<6} {x_novo:<15.10f} {f(x_novo):<15.10e} {erro:<15.10e}")
        
        # Verifica convergência
        if erro < tol:
            print(f"\nConvergiu em {n} iterações!")
            print(f"Raiz encontrada: x = {x_novo:.12f}")
            print(f"f(x) = {f(x_novo):.2e}")
            break
        
        x_atual = x_novo
    
    return np.array(iteracoes), np.array(erros), np.array(residuos)


def verificar_condicoes_banach(f, df, ddf, a, b):
    """
    Verifica as condições da Proposição 3.6 para garantir
    que o Método de Newton converge pelo Princípio de Banach.
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DAS CONDIÇÕES DO TEOREMA")
    print("=" * 60)
    
    # Pontos de teste no intervalo
    x_test = np.linspace(a, b, 1000)
    
    # Condição (3.4.1): |f(x)f''(x)/(f'(x))²| ≤ λ < 1
    razao = np.abs(f(x_test) * ddf(x_test) / (df(x_test)**2))
    lambda_estimado = np.max(razao)
    
    print(f"\nIntervalo: [{a}, {b}]")
    print(f"Condição (3.4.1): |f(x)f''(x)/(f'(x))²| ≤ λ")
    print(f"λ estimado = {lambda_estimado:.6f}")
    
    if lambda_estimado < 1:
        print("✓ Condição satisfeita (λ < 1) - F é uma contração")
    else:
        print("✗ Condição NÃO satisfeita (λ ≥ 1)")
    
    # Condição (3.4.2): |f(x̄)/f'(x̄)| ≤ (1-λ)ζ
    x_bar = (a + b) / 2
    zeta = (b - a) / 2
    razao_bar = abs(f(x_bar) / df(x_bar))
    limite = (1 - lambda_estimado) * zeta
    
    print(f"\nCondição (3.4.2): |f(x̄)/f'(x̄)| ≤ (1-λ)ζ")
    print(f"x̄ = {x_bar:.6f}")
    print(f"|f(x̄)/f'(x̄)| = {razao_bar:.6f}")
    print(f"(1-λ)ζ = {limite:.6f}")
    
    if razao_bar <= limite:
        print("✓ Condição satisfeita - F([a,b]) ⊂ [a,b]")
    else:
        print("✗ Condição NÃO satisfeita")
    
    print("=" * 60)
    
    return lambda_estimado < 1 and razao_bar <= limite


# Exemplo 1: Raiz quadrada de 2
def exemplo1_newton():
    """
    Encontra √2 resolvendo f(x) = x² - 2 = 0
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 1: Cálculo de √2")
    print("Resolver: f(x) = x² - 2 = 0")
    print("=" * 70)
    
    # Define função e derivadas
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    ddf = lambda x: 2
    
    # Aproximação inicial
    x0 = 1.5
    
    # Verifica condições de Banach
    a, b = 1.0, 2.0
    converge = verificar_condicoes_banach(f, df, ddf, a, b)
    
    # Executa o método de Newton
    print("\n" + "-" * 70)
    print("ITERAÇÕES DO MÉTODO DE NEWTON")
    print("-" * 70)
    iteracoes, erros, residuos = metodo_newton_banach(f, df, ddf, x0)
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Gráfico da função
    x_plot = np.linspace(0.5, 2.5, 500)
    axes[0, 0].plot(x_plot, f(x_plot), 'b-', linewidth=2, label='f(x) = x² - 2')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=np.sqrt(2), color='r', linestyle='--', alpha=0.5, 
                       label=f'√2 = {np.sqrt(2):.6f}')
    axes[0, 0].plot(iteracoes, f(iteracoes), 'ro', markersize=8, label='Iterações')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title('Função f(x) = x² - 2')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Visualização geométrica do método de Newton
    axes[0, 1].plot(x_plot, f(x_plot), 'b-', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Desenha as tangentes
    for i in range(min(5, len(iteracoes)-1)):
        xi = iteracoes[i]
        yi = f(xi)
        # Tangente: y - f(xi) = f'(xi)(x - xi)
        x_tang = np.linspace(xi - 0.5, xi + 0.5, 100)
        y_tang = yi + df(xi) * (x_tang - xi)
        axes[0, 1].plot(x_tang, y_tang, 'g--', alpha=0.5)
        axes[0, 1].plot(xi, yi, 'ro', markersize=8)
        axes[0, 1].plot([xi, iteracoes[i+1]], [yi, 0], 'r:', alpha=0.5)
    
    axes[0, 1].set_xlim([1, 2])
    axes[0, 1].set_ylim([-1, 1])
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    axes[0, 1].set_title('Interpretação Geométrica do Método de Newton')
    axes[0, 1].grid(True)
    
    # 3. Convergência dos erros
    axes[1, 0].semilogy(range(1, len(erros)+1), erros, 'bo-', linewidth=2, 
                        markersize=6, label='Erro |x_{n+1} - x_n|')
    axes[1, 0].semilogy(range(1, len(residuos)+1), residuos, 'rs-', linewidth=2, 
                        markersize=6, label='Resíduo |f(x_n)|')
    axes[1, 0].set_xlabel('Iteração')
    axes[1, 0].set_ylabel('Erro/Resíduo (escala log)')
    axes[1, 0].set_title('Convergência Quadrática do Método de Newton')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Sequência de aproximações
    axes[1, 1].plot(range(len(iteracoes)), iteracoes, 'go-', linewidth=2, 
                    markersize=8, label='Iterações x_n')
    axes[1, 1].axhline(y=np.sqrt(2), color='r', linestyle='--', linewidth=2, 
                      label=f'√2 = {np.sqrt(2):.10f}')
    axes[1, 1].set_xlabel('Iteração n')
    axes[1, 1].set_ylabel('x_n')
    axes[1, 1].set_title('Sequência de Aproximações')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('newton_exemplo1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return iteracoes


# Exemplo 2: Raiz de polinômio
def exemplo2_newton():
    """
    Encontra raiz de f(x) = x³ - x - 1 = 0
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Raiz de Polinômio Cúbico")
    print("Resolver: f(x) = x³ - x - 1 = 0")
    print("=" * 70)
    
    # Define função e derivadas
    f = lambda x: x**3 - x - 1
    df = lambda x: 3*x**2 - 1
    ddf = lambda x: 6*x
    
    # Aproximação inicial
    x0 = 1.5
    
    # Verifica condições
    a, b = 1.0, 2.0
    converge = verificar_condicoes_banach(f, df, ddf, a, b)
    
    # Executa o método
    print("\n" + "-" * 70)
    print("ITERAÇÕES DO MÉTODO DE NEWTON")
    print("-" * 70)
    iteracoes, erros, residuos = metodo_newton_banach(f, df, ddf, x0)
    
    # Visualização
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Função
    x_plot = np.linspace(0, 2, 500)
    axes[0].plot(x_plot, f(x_plot), 'b-', linewidth=2, label='f(x) = x³ - x - 1')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].axvline(x=iteracoes[-1], color='r', linestyle='--', alpha=0.5, 
                    label=f'Raiz ≈ {iteracoes[-1]:.6f}')
    axes[0].plot(iteracoes, f(iteracoes), 'ro', markersize=8, label='Iterações')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('f(x) = x³ - x - 1')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Taxa de convergência
    if len(erros) > 1:
        razao = erros[1:] / (erros[:-1]**2)  # e_{n+1} / e_n²
        axes[1].plot(range(1, len(razao)+1), razao, 'mo-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Iteração')
        axes[1].set_ylabel('e_{n+1} / e_n²')
        axes[1].set_title('Verificação da Convergência Quadrática')
        axes[1].grid(True)
    
    # 3. Erro vs Resíduo
    axes[2].loglog(residuos, erros[:-1] if len(erros) > len(residuos) else erros, 
                   'co-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Resíduo |f(x_n)|')
    axes[2].set_ylabel('Erro |x_{n+1} - x_n|')
    axes[2].set_title('Relação Erro-Resíduo')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('newton_exemplo2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return iteracoes


# Exemplo 3: Equação transcendental
def exemplo3_newton():
    """
    Encontra raiz de f(x) = e^x - 3x = 0
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Equação Transcendental")
    print("Resolver: f(x) = e^x - 3x = 0")
    print("=" * 70)
    
    # Define função e derivadas
    f = lambda x: np.exp(x) - 3*x
    df = lambda x: np.exp(x) - 3
    ddf = lambda x: np.exp(x)
    
    # Existem duas raízes - vamos encontrar ambas com diferentes x0
    x0_list = [0.5, 1.5]
    raizes = []
    
    for i, x0 in enumerate(x0_list):
        print(f"\n{'='*70}")
        print(f"Buscando raiz {i+1} com x0 = {x0}")
        print(f"{'='*70}")
        
        iteracoes, erros, residuos = metodo_newton_banach(f, df, ddf, x0, max_iter=15)
        raizes.append(iteracoes[-1])
    
    # Visualização
    plt.figure(figsize=(14, 5))
    
    # 1. Gráfico mostrando ambas as raízes
    plt.subplot(1, 2, 1)
    x_plot = np.linspace(-0.5, 2, 500)
    plt.plot(x_plot, f(x_plot), 'b-', linewidth=2, label='f(x) = e^x - 3x')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    for i, raiz in enumerate(raizes):
        plt.axvline(x=raiz, color=f'C{i+1}', linestyle='--', alpha=0.7, 
                    label=f'Raiz {i+1} ≈ {raiz:.6f}')
        plt.plot(raiz, f(raiz), 'o', color=f'C{i+1}', markersize=10)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = e^x - 3x e suas Raízes')
    plt.legend()
    plt.grid(True)
    
    # 2. Convergência para cada raiz
    plt.subplot(1, 2, 2)
    for i, x0 in enumerate(x0_list):
        iteracoes, erros, residuos = metodo_newton_banach(f, df, ddf, x0, max_iter=15)
        plt.semilogy(range(1, len(erros)+1), erros, 'o-', linewidth=2, 
                    markersize=6, label=f'Raiz {i+1} (x0={x0})')
    plt.xlabel('Iteração')
    plt.ylabel('Erro (escala log)')
    plt.title('Convergência para Diferentes Raízes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('newton_exemplo3.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return raizes


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "MÉTODO DE NEWTON")
    print(" " * 15 + "Via Princípio da Contração de Banach")
    print(" " * 20 + "(Proposição 3.6)")
    print("=" * 80)
    
    # Executa os exemplos
    raiz1 = exemplo1_newton()
    raiz2 = exemplo2_newton()
    raizes3 = exemplo3_newton()
    
    print("\n" + "=" * 80)
    print(" " * 25 + "Todos os exemplos executados!")
    print("=" * 80)
    
    print("\nRESUMO DOS RESULTADOS:")
    print(f"  Exemplo 1 (√2):        {raiz1[-1]:.12f}")
    print(f"  Exemplo 2 (x³-x-1=0):  {raiz2[-1]:.12f}")
    print(f"  Exemplo 3a (e^x-3x=0): {raizes3[0]:.12f}")
    print(f"  Exemplo 3b (e^x-3x=0): {raizes3[1]:.12f}")
