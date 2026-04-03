import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Optimization Portfolio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------- SIDEBAR PROFILE -----------
with st.sidebar:
    st.markdown("## 👨‍💻 G. Sathwik Reddy")
    st.markdown("**Optimization Algorithm Visualizer**")
    st.divider()
    
    st.markdown("### Profile")
    st.write("**🎓 Institution:** KLH University")
    st.write("**📚 Department:** ECE (Electronics & Communication)")
    
    st.divider()
    st.markdown("### Expertise")
    st.write("• Optimization Algorithms")
    st.write("• Machine Learning & AI")
    st.write("• Data Visualization")
    
    st.divider()
    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🐍 Python")
        st.write("🧮 NumPy")
    with col2:
        st.write("📊 Matplotlib")
        st.write("⚡ Streamlit")
    
    st.divider()
    st.markdown("### Portfolio")
    st.write("📈 4 Optimization Algorithms")
    st.write("🎯 Interactive Visualizations")
    st.write("⚡ Dashboard UI")
    
    st.markdown("---")
    st.markdown("""
    <p style="color: #707080; font-size: 0.8rem; text-align: center;">
    Built for Optimization Algorithms Project
    </p>
    """, unsafe_allow_html=True)

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
.title {
    font-size: 3rem;
    font-weight: bold;
    color: #00e6cc;
}
.subtitle {
    color: #00bfa6;
    font-size: 1.2rem;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #111827;
    border: 1px solid #00e6cc;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 0.85rem;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ----------- HEADER -----------
st.markdown('<p class="title">🚀 Optimization Portfolio</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Algorithm Visualizer & Performance Analyzer<br>By G. Sathwik Reddy</p>', unsafe_allow_html=True)

st.markdown("---")

# ----------- TABS -----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", 
    "📉 Unconstrained Minimization", 
    "🎯 Pareto Front",
    "🧬 Genetic Algorithm",
    "🔥 Simulated Annealing"
])

# ----------- HOME TAB -----------
with tab1:
    st.header("Welcome to Optimization Algorithm Visualizer")
    
    st.success("This dashboard demonstrates Gradient Descent, Pareto Optimization, Genetic Algorithms, and Simulated Annealing with interactive visualizations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📉 Gradient-Based Methods</h3>
        <p>Unconstrained minimization using gradient descent and optimization techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🧬 Metaheuristic Algorithms</h3>
        <p>Genetic Algorithms, Simulated Annealing, and Pareto Optimization.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------- UNCONSTRAINED TAB -----------
with tab2:
    st.header("Unconstrained Minimization - Gradient Descent")

    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    start = st.slider("Start Point", -10.0, 10.0, 5.0)
    iterations = st.slider("Iterations", 10, 200, 50)

    def f(x):
        return x**2 + 4*x + 4

    def grad(x):
        return 2*x + 4

    if st.button("Run Optimization"):
        x_vals = np.linspace(-10, 10, 200)
        y_vals = f(x_vals)

        x = start
        history = [x]
        loss_history = [f(x)]

        for _ in range(iterations):
            x = x - lr * grad(x)
            history.append(x)
            loss_history.append(f(x))

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.scatter(history, loss_history, color='red')
        ax.set_title("Optimization Path")

        st.pyplot(fig)

# ----------- PARETO TAB -----------
with tab3:
    st.header("Pareto Front Visualization")

    pop_size = st.slider("Population Size", 10, 100, 50)
    generations = st.slider("Generations", 10, 100, 30)

    if st.button("Generate Pareto Front"):
        population = np.random.uniform(-5, 5, (pop_size, 2))

        def f1(x, y):
            return (x - 1)**2 + (y - 1)**2

        def f2(x, y):
            return (x + 1)**2 + (y + 1)**2

        for g in range(generations):
            population += np.random.normal(0, 0.1, population.shape)

        f1_vals = f1(population[:,0], population[:,1])
        f2_vals = f2(population[:,0], population[:,1])

        fig, ax = plt.subplots()
        ax.scatter(f1_vals, f2_vals)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")

        st.pyplot(fig)

# ----------- GENETIC ALGORITHM TAB -----------
with tab4:
    st.header("Genetic Algorithm")

    pop_size = st.slider("Population Size (GA)", 20, 200, 100)
    generations = st.slider("Generations (GA)", 10, 200, 50)
    mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)

    def fitness(x):
        return 1 / (1 + np.sum(x**2))

    if st.button("Run Genetic Algorithm"):
        population = np.random.uniform(-5, 5, (pop_size, 2))
        best_history = []

        for g in range(generations):
            fit = np.array([fitness(ind) for ind in population])
            best_history.append(np.max(fit))

            probs = fit / np.sum(fit)
            idx = np.random.choice(pop_size, pop_size, p=probs)
            population = population[idx]
            population += np.random.normal(0, mutation_rate, population.shape)

        fig, ax = plt.subplots()
        ax.plot(best_history)
        ax.set_title("GA Convergence")

        st.pyplot(fig)

# ----------- SIMULATED ANNEALING TAB -----------
with tab5:
    st.header("Simulated Annealing")

    temp = st.slider("Initial Temperature", 100, 1000, 500)
    cooling = st.slider("Cooling Rate", 0.90, 0.99, 0.95)

    def cost(x):
        return x**2

    if st.button("Run Simulated Annealing"):
        x = np.random.uniform(-10, 10)
        best = x
        best_cost = cost(x)

        costs = []
        temperature = temp

        while temperature > 1:
            new_x = x + np.random.normal(0, 1)
            if cost(new_x) < cost(x) or np.random.rand() < np.exp(-(cost(new_x)-cost(x))/temperature):
                x = new_x

            if cost(x) < best_cost:
                best = x
                best_cost = cost(x)

            costs.append(best_cost)
            temperature *= cooling

        fig, ax = plt.subplots()
        ax.plot(costs)
        ax.set_title("Simulated Annealing Convergence")

        st.pyplot(fig)

# ----------- FOOTER -----------
st.markdown("""
<div class="footer">
👨‍💻 <b>G. Sathwik Reddy</b> | Optimization Algorithm Visualizer<br>
KLH University • Electronics & Communication Engineering
</div>
""", unsafe_allow_html=True)
