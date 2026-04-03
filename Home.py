import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Optimization Algorithm Visualizer",
    layout="wide",
    page_icon="🚀"
)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("⚙️ Controls")
    algorithm = st.selectbox(
        "Select Algorithm",
        ["Unconstrained Minimization",
         "Pareto Front",
         "Genetic Algorithm",
         "Simulated Annealing"]
    )

    iterations = st.slider("Iterations", 10, 200, 50)
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    st.markdown("---")
    st.info("Adjust parameters and run the algorithm.")

# ------------------ HEADER ------------------
st.title("🚀 Optimization Algorithm Visualizer")
st.markdown("Interactive visualization of optimization algorithms")
st.markdown("---")

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Unconstrained",
    "🎯 Pareto Front",
    "🧬 Genetic Algorithm",
    "🔥 Simulated Annealing"
])

# =====================================================
# TAB 1 — UNCONSTRAINED MINIMIZATION
# =====================================================
with tab1:
    st.subheader("Gradient Descent Optimization")

    start = st.slider("Starting Point", -10.0, 10.0, 5.0)

    def f(x):
        return x**2 + 4*x + 4

    def grad(x):
        return 2*x + 4

    if st.button("Run Gradient Descent"):
        x_vals = np.linspace(-10, 10, 200)
        y_vals = f(x_vals)

        x = start
        history = [x]
        loss_history = [f(x)]

        progress = st.progress(0)
        plot_placeholder = st.empty()

        for i in range(iterations):
            g = grad(x)
            x = x - learning_rate * g
            history.append(x)
            loss_history.append(f(x))

            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals)
            ax.scatter(history, loss_history, color="red")
            ax.set_title("Optimization Path")

            plot_placeholder.pyplot(fig)
            progress.progress((i + 1) / iterations)
            time.sleep(0.05)

        col1, col2, col3 = st.columns(3)
        col1.metric("Final x", round(x, 4))
        col2.metric("Final Loss", round(loss_history[-1], 4))
        col3.metric("Iterations", iterations)

# =====================================================
# TAB 2 — PARETO FRONT
# =====================================================
with tab2:
    st.subheader("Pareto Front Visualization")

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
        ax.set_title("Pareto Front")

        st.pyplot(fig)

# =====================================================
# TAB 3 — GENETIC ALGORITHM
# =====================================================
with tab3:
    st.subheader("Genetic Algorithm Optimization")

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
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")

        st.pyplot(fig)

# =====================================================
# TAB 4 — SIMULATED ANNEALING
# =====================================================
with tab4:
    st.subheader("Simulated Annealing")

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
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")

        st.pyplot(fig)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Optimization Algorithm Visualizer • Streamlit Dashboard")
