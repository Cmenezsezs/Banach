import numpy as np
import matplotlib.pyplot as plt

def equacao_fredholm(f, K, lambda_val, a, b, max_iter=50, tol=1e-6):
    """
    Resolve a equação integral de Fredholm:
    u(x) = f(x) + λ * ∫[a,b] K(x,y,u(y)) dy
    
    Usando o método das aproximações sucessivas (Princípio de Banach).
    
    Parâmetros:
    - f: função contínua f(x)
    - K: núcleo K(x, y, u) - função de três variáveis
    - lambda_val: parâmetro λ
    - a, b: limites do intervalo [a,b]
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência
    """
    
    # Pontos de discretização
    n_points = 100
    x = np.linspace(a, b, n_points)
    
    # Aproximação inicial u_0(x) = f(x)
    u_atual = f(x)
    
    # Iterações do método
    historico = [u_atual.copy()]
    
    for iteracao in range(max_iter):
        u_novo = np.zeros_like(x)
        
        # Para cada ponto x_i
        for i, xi in enumerate(x):
            # Calcula a integral usando regra do trapézio
            integral = 0
            y = x  # Pontos de integração
            
            # Calcula K(xi, y, u(y)) para todos os y
            K_valores = np.array([K(xi, yj, u_atual[j]) for j, yj in enumerate(y)])
            
            # Integração numérica (regra do trapézio)
            integral = np.trapz(K_valores, y)
            
            # Aplica a fórmula u_{n+1}(x) = f(x) + λ * integral
            u_novo[i] = f(xi) + lambda_val * integral
        
        # Verifica convergência
        erro = np.max(np.abs(u_novo - u_atual))
        
        if erro < tol:
            print(f"Convergiu em {iteracao + 1} iterações com erro {erro:.2e}")
            return x, u_novo, historico
        
        u_atual = u_novo.copy()
        historico.append(u_atual.copy())
    
    print(f"Atingiu o máximo de iterações ({max_iter})")
    return x, u_atual, historico


# Exemplo 1: Núcleo simples K(x,y,u) = x*y*u
def exemplo1():
    """
    Exemplo com núcleo linear K(x,y,u) = x*y*u
    """
    print("\n=== EXEMPLO 1: Equação de Fredholm ===")
    print("u(x) = sin(x) + λ * ∫[0,1] x*y*u(y) dy\n")
    
    # Define as funções
    f = lambda x: np.sin(x)
    K = lambda x, y, u: x * y * u
    
    # Parâmetros
    lambda_val = 0.1  # Deve ser pequeno para garantir contração
    a, b = 0, 1
    
    # Resolve
    x, u_solucao, historico = equacao_fredholm(f, K, lambda_val, a, b)
    
    # Visualização
    plt.figure(figsize=(12, 5))
    
    # Solução final
    plt.subplot(1, 2, 1)
    plt.plot(x, f(x), 'b--', label='f(x) = sin(x)', linewidth=2)
    plt.plot(x, u_solucao, 'r-', label='u(x) - Solução', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solução da Equação de Fredholm')
    plt.legend()
    plt.grid(True)
    
    # Convergência das iterações
    plt.subplot(1, 2, 2)
    for i, u in enumerate(historico[::5]):  # Mostra a cada 5 iterações
        plt.plot(x, u, alpha=0.5, label=f'Iteração {i*5}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Convergência das Aproximações Sucessivas')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fredholm_exemplo1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return x, u_solucao


# Exemplo 2: Núcleo não-linear
def exemplo2():
    """
    Exemplo com núcleo não-linear mais complexo
    """
    print("\n=== EXEMPLO 2: Equação de Fredholm Não-Linear ===")
    print("u(x) = x² + λ * ∫[0,π] sin(x+y)*u(y)/(1+u(y)²) dy\n")
    
    # Define as funções
    f = lambda x: x**2
    K = lambda x, y, u: np.sin(x + y) * u / (1 + u**2)
    
    # Parâmetros
    lambda_val = 0.05  # Pequeno para garantir contração
    a, b = 0, np.pi
    
    # Resolve
    x, u_solucao, historico = equacao_fredholm(f, K, lambda_val, a, b)
    
    # Visualização
    plt.figure(figsize=(10, 6))
    plt.plot(x, f(x), 'b--', label='f(x) = x²', linewidth=2)
    plt.plot(x, u_solucao, 'r-', label='u(x) - Solução', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solução da Equação de Fredholm Não-Linear')
    plt.legend()
    plt.grid(True)
    plt.savefig('fredholm_exemplo2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return x, u_solucao


if __name__ == "__main__":
    print("=" * 60)
    print("EQUAÇÕES INTEGRAIS DE FREDHOLM")
    print("Método das Aproximações Sucessivas")
    print("Baseado no Princípio da Contração de Banach")
    print("=" * 60)
    
    # Executa os exemplos
    x1, u1 = exemplo1()
    x2, u2 = exemplo2()
    
    print("\n" + "=" * 60)
    print("Programas executados com sucesso!")
    print("=" * 60)
