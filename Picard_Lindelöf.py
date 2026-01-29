import numpy as np
import matplotlib.pyplot as plt

def teorema_picard_lindelof(f, t0, x0, a, M, max_iter=20, n_points=100):
    """
    Resolve o problema de valor inicial (PVI):
    dx/dt = f(t, x)
    x(t0) = x0
    
    Usando o método de Picard (aproximações sucessivas).
    
    Parâmetros:
    - f: função f(t, x) do lado direito da EDO
    - t0: tempo inicial
    - x0: condição inicial x(t0) = x0
    - a: raio do intervalo [t0-a, t0+a]
    - M: constante tal que |f(t,x)| ≤ M
    - max_iter: número de iterações de Picard
    - n_points: número de pontos para discretização
    """
    
    # Intervalo de solução
    alpha = a  # Para simplificar
    t = np.linspace(t0 - alpha, t0 + alpha, n_points)
    
    # Aproximações de Picard
    aproximacoes = []
    
    # Aproximação inicial: φ_0(t) = x0
    x_atual = np.ones_like(t) * x0
    aproximacoes.append(x_atual.copy())
    
    print(f"Iteração 0: φ_0(t) = {x0}")
    
    # Iterações de Picard
    for n in range(1, max_iter + 1):
        x_novo = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            # φ_{n+1}(t) = x0 + ∫[t0, t] f(s, φ_n(s)) ds
            
            # Pontos de integração de t0 até ti
            if ti >= t0:
                t_int = np.linspace(t0, ti, 50)
            else:
                t_int = np.linspace(ti, t0, 50)
            
            # Interpola x_atual para os pontos de integração
            x_int = np.interp(t_int, t, x_atual)
            
            # Calcula f(s, φ_n(s))
            f_valores = np.array([f(s, x_int[j]) for j, s in enumerate(t_int)])
            
            # Integra usando regra do trapézio
            if ti >= t0:
                integral = np.trapz(f_valores, t_int)
            else:
                integral = -np.trapz(f_valores, t_int)
            
            x_novo[i] = x0 + integral
        
        # Calcula o erro
        erro = np.max(np.abs(x_novo - x_atual))
        print(f"Iteração {n}: erro = {erro:.6e}")
        
        x_atual = x_novo.copy()
        aproximacoes.append(x_atual.copy())
        
        if erro < 1e-8:
            print(f"\nConvergência atingida em {n} iterações!")
            break
    
    return t, aproximacoes


# Exemplo 1: EDO linear simples
def exemplo1_edo():
    """
    Exemplo: dx/dt = x, x(0) = 1
    Solução exata: x(t) = e^t
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 1: EDO Linear")
    print("dx/dt = x, x(0) = 1")
    print("Solução exata: x(t) = e^t")
    print("=" * 60 + "\n")
    
    # Define a EDO
    f = lambda t, x: x
    t0 = 0
    x0 = 1
    a = 1.0
    M = np.e  # |f| ≤ e no intervalo
    
    # Resolve usando Picard
    t, aproximacoes = teorema_picard_lindelof(f, t0, x0, a, M, max_iter=10)
    
    # Solução exata
    x_exato = np.exp(t)
    
    # Visualização
    plt.figure(figsize=(14, 5))
    
    # Comparação das iterações
    plt.subplot(1, 2, 1)
    cores = plt.cm.viridis(np.linspace(0, 1, len(aproximacoes)))
    for i, aprox in enumerate(aproximacoes):
        if i % 2 == 0 or i == len(aproximacoes) - 1:  # Mostra iterações pares e última
            plt.plot(t, aprox, color=cores[i], alpha=0.7, 
                    label=f'φ_{i}(t)', linewidth=2)
    
    plt.plot(t, x_exato, 'k--', label='Solução exata: e^t', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Aproximações de Picard')
    plt.legend()
    plt.grid(True)
    
    # Erro em relação à solução exata
    plt.subplot(1, 2, 2)
    for i, aprox in enumerate(aproximacoes):
        erro = np.abs(aprox - x_exato)
        if i % 2 == 0 or i == len(aproximacoes) - 1:
            plt.semilogy(t, erro, color=cores[i], alpha=0.7, 
                        label=f'Erro φ_{i}(t)', linewidth=2)
    
    plt.xlabel('t')
    plt.ylabel('Erro absoluto (escala log)')
    plt.title('Convergência para a Solução Exata')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('picard_exemplo1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcula erro final
    erro_final = np.max(np.abs(aproximacoes[-1] - x_exato))
    print(f"\nErro final: {erro_final:.6e}")
    
    return t, aproximacoes, x_exato


# Exemplo 2: EDO não-linear
def exemplo2_edo():
    """
    Exemplo: dx/dt = t + x², x(0) = 0
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 2: EDO Não-Linear")
    print("dx/dt = t + x², x(0) = 0")
    print("=" * 60 + "\n")
    
    # Define a EDO
    f = lambda t, x: t + x**2
    t0 = 0
    x0 = 0
    a = 0.5  # Intervalo menor para garantir convergência
    M = 2.0
    
    # Resolve usando Picard
    t, aproximacoes = teorema_picard_lindelof(f, t0, x0, a, M, max_iter=15)
    
    # Visualização
    plt.figure(figsize=(12, 5))
    
    # Aproximações de Picard
    plt.subplot(1, 2, 1)
    cores = plt.cm.plasma(np.linspace(0, 1, len(aproximacoes)))
    for i, aprox in enumerate(aproximacoes):
        if i % 2 == 0 or i == len(aproximacoes) - 1:
            plt.plot(t, aprox, color=cores[i], alpha=0.7, 
                    label=f'φ_{i}(t)', linewidth=2)
    
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Aproximações de Picard para EDO Não-Linear')
    plt.legend()
    plt.grid(True)
    
    # Diferença entre iterações consecutivas
    plt.subplot(1, 2, 2)
    for i in range(1, len(aproximacoes)):
        diferenca = np.abs(aproximacoes[i] - aproximacoes[i-1])
        if i % 2 == 0 or i == len(aproximacoes) - 1:
            plt.semilogy(t, diferenca, color=cores[i], alpha=0.7, 
                        label=f'|φ_{i} - φ_{i-1}|', linewidth=2)
    
    plt.xlabel('t')
    plt.ylabel('Diferença (escala log)')
    plt.title('Convergência das Aproximações')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('picard_exemplo2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return t, aproximacoes


# Exemplo 3: Sistema de EDOs
def exemplo3_sistema():
    """
    Exemplo: Sistema de EDOs
    dx/dt = y
    dy/dt = -x
    x(0) = 0, y(0) = 1
    Solução exata: x(t) = sin(t), y(t) = cos(t)
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Sistema de EDOs")
    print("dx/dt = y,  dy/dt = -x")
    print("x(0) = 0, y(0) = 1")
    print("Solução exata: x(t) = sin(t), y(t) = cos(t)")
    print("=" * 60 + "\n")
    
    # Para sistema, adaptar para vetor
    def f_sistema(t, X):
        x, y = X
        return np.array([y, -x])
    
    t0 = 0
    X0 = np.array([0, 1])  # [x0, y0]
    a = np.pi/2
    
    # Intervalo
    t = np.linspace(t0 - a, t0 + a, 100)
    
    # Aproximações
    aproximacoes_x = []
    aproximacoes_y = []
    
    # Inicial
    X_atual = np.tile(X0, (len(t), 1))
    aproximacoes_x.append(X_atual[:, 0].copy())
    aproximacoes_y.append(X_atual[:, 1].copy())
    
    # Iterações
    for n in range(1, 10):
        X_novo = np.zeros_like(X_atual)
        
        for i, ti in enumerate(t):
            if ti >= t0:
                t_int = np.linspace(t0, ti, 50)
            else:
                t_int = np.linspace(ti, t0, 50)
            
            # Interpola
            X_int = np.zeros((len(t_int), 2))
            X_int[:, 0] = np.interp(t_int, t, X_atual[:, 0])
            X_int[:, 1] = np.interp(t_int, t, X_atual[:, 1])
            
            # Calcula f
            f_valores = np.array([f_sistema(s, X_int[j]) for j, s in enumerate(t_int)])
            
            # Integra
            if ti >= t0:
                integral = np.trapz(f_valores, t_int, axis=0)
            else:
                integral = -np.trapz(f_valores, t_int, axis=0)
            
            X_novo[i] = X0 + integral
        
        X_atual = X_novo.copy()
        aproximacoes_x.append(X_atual[:, 0].copy())
        aproximacoes_y.append(X_atual[:, 1].copy())
    
    # Soluções exatas
    x_exato = np.sin(t)
    y_exato = np.cos(t)
    
    # Visualização
    plt.figure(figsize=(14, 5))
    
    # Componente x(t)
    plt.subplot(1, 2, 1)
    cores = plt.cm.cool(np.linspace(0, 1, len(aproximacoes_x)))
    for i, aprox in enumerate(aproximacoes_x):
        if i % 2 == 0 or i == len(aproximacoes_x) - 1:
            plt.plot(t, aprox, color=cores[i], alpha=0.7, 
                    label=f'x_{i}(t)', linewidth=2)
    plt.plot(t, x_exato, 'k--', label='x(t) = sin(t)', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Componente x(t)')
    plt.legend()
    plt.grid(True)
    
    # Espaço de fases
    plt.subplot(1, 2, 2)
    for i in range(len(aproximacoes_x)):
        if i % 2 == 0 or i == len(aproximacoes_x) - 1:
            plt.plot(aproximacoes_x[i], aproximacoes_y[i], color=cores[i], 
                    alpha=0.7, label=f'Iter {i}', linewidth=2)
    plt.plot(x_exato, y_exato, 'k--', label='Exato', linewidth=2)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.title('Espaço de Fases')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('picard_sistema.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return t, aproximacoes_x, aproximacoes_y


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "TEOREMA DE PICARD-LINDELÖF")
    print(" " * 10 + "Existência e Unicidade de Soluções de EDOs")
    print(" " * 10 + "Método das Aproximações Sucessivas de Picard")
    print("=" * 70)
    
    # Executa os exemplos
    t1, aprox1, exato1 = exemplo1_edo()
    t2, aprox2 = exemplo2_edo()
    t3, aprox_x3, aprox_y3 = exemplo3_sistema()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "Programas executados com sucesso!")
    print("=" * 70)
