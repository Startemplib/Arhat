import matplotlib.pyplot as plt

# Adjust figure size to fit the text size (narrow rectangle)
fig = plt.figure(figsize=(4.53, 1.3137), facecolor="black")

# Create a plot and add the text in white with Times New Roman italic
plt.text(
    0.5,
    0.4,
    """"Breaking through the Empyrean."
    
    —— Arhat is here
    """,
    fontsize=12,
    fontfamily="Times New Roman",
    fontstyle="italic",
    color="white",
    ha="center",
    va="center",
)

# Hide the axes
plt.gca().set_axis_off()

# Show the plot with the styled text
plt.show(block=False)  # Non-blocking display

# Pause the display for 5 seconds
plt.pause(1.3)

# Close the plot after the pause
plt.close()
