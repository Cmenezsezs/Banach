# O Princípio da Contração de Banach e Aplicações

Este repositório contém a documentação teórica e implementações práticas baseadas no Trabalho de Conclusão de Curso **"O Princípio da Contração de Banach"**, de Clemerson Oliveira da Silva Menezes (UFAL, 2013).

O projeto explora a fundamentação dos Espaços Métricos Completos e demonstra como o Princípio do Ponto Fixo de Banach é a base para resolver problemas complexos em Equações Diferenciais, Integrais e Cálculo Numérico.

---

## 📚 Conteúdo Teórico (Resumo do PDF)

O documento base está estruturado em três pilares fundamentais:

### 1. Fundamentação Topológica
Introdução aos conceitos necessários para a compreensão do princípio:
- **Espaços Métricos e Topológicos:** Definições, bolas abertas/fechadas, conjuntos abertos/fechados.
- **Convergência e Continuidade:** Diferença entre continuidade pontual e uniforme.
- **Espaços Métricos Completos:** Sequências de Cauchy e a importância da completude (ex: $\mathbb{R}$ e $\mathbb{R}^n$).

### 2. O Princípio da Contração de Banach
O coração do trabalho, que estabelece:
> *"Toda contração definida em um espaço métrico completo admite um único ponto fixo."*

O texto apresenta a demonstração formal e o método construtivo das **Aproximações Sucessivas**:
$$x_{n+1} = T(x_n)$$

### 3. Aplicações Matemáticas
O trabalho detalha quatro grandes aplicações do princípio:
1.  **Equações Integrais de Fredholm:** Existência e unicidade de soluções para integrais do tipo $u(x) = f(x) + \lambda \int K(x,y,u(y)) dy$.
2.  **Teorema de Picard-Lindelöf:** Garantia de solução única para Problemas de Valor Inicial (PVI) em EDOs.
3.  **Teorema de Stampacchia:** Aplicação em problemas variacionais (embora não implementado em código neste repo).
4.  **Método de Newton:** Uma abordagem via ponto fixo para encontrar zeros de funções reais.

---

## 💻 Implementações em Python

Este projeto inclui scripts Python que traduzem a teoria para a prática numérica.

### 📁 Estrutura de Arquivos

```text
├── bases_de_dados/          # Arquivos CSV gerados com os resultados
│   ├── fredholm_data.csv    # Dados da equação integral
│   ├── picard_data.csv      # Iterações da solução da EDO
│   └── newton_data.csv      # Convergência do método de Newton
├── scripts/
│   ├── app_fredholm.py      # Solução de Equações Integrais
│   ├── app_picard.py        # Solver de EDOs via Picard
│   └── app_newton.py        # Método de Newton via Banach
└── README.md
