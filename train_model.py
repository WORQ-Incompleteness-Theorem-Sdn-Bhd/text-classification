import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Example dataset
data = {
    "text": [
            "I truly appreciate your responsibility in ensuring that every project is completed on time. Your dedication does not go unnoticed.",
        "Thank you for your integrity in handling every task with transparency. Your honesty sets a great example for everyone on the team.",
        "Your reliability has made a significant difference in our projects. We can always count on you to get things done right.",
        "I admire how dependable you are, always ready to step in and help when needed. You're an asset to the team.",
        "Your ownership of tasks is something I deeply respect. You take full responsibility for the work, and it always shows in the results.",
        "I appreciate your commitment to the team and our goals. Your hard work and consistency are truly inspiring.",
        "Your diligence is evident in everything you do. You ensure no detail is missed, and that's why your work is always top-notch.",
        "Thank you for your transparency in communication. It makes working together so much smoother and more effective.",
        "Your initiative in taking on new challenges has not gone unnoticed. You consistently step up and make things happen.",
        "I admire your conscientiousness in every decision you make. You always consider how it affects the team and work accordingly.",
        "Your accountability is truly remarkable. You take responsibility for your actions and always follow through with excellence.",
        "Your punctuality and consistent work ethic show a high level of reliability. You're always on top of your tasks, and that’s inspiring.",
        "Thank you for always taking ownership of your work. It's clear you have a genuine commitment to delivering the best results.",
        "Your attention to detail and your ability to follow through are key to our team's success. You are a model of accountability.",
        "I’m grateful for how you handle every project with such responsibility. Your commitment to excellence is a big part of why we succeed.",
        "Your dependability is one of the reasons our team functions so smoothly. I can always trust that you'll get the job done right.",
        "I appreciate your transparency in sharing challenges and solutions. It encourages openness and helps us work together more effectively.",
        "Your initiative in leading by example and taking responsibility for new tasks is incredibly motivating for the team.",
        "Your conscientiousness and careful approach to every task make you a valuable team member. We can always rely on your thoroughness.",
        "Thank you for being so accountable in everything you do. Your consistent follow-through sets a standard for all of us.",
         "Accountability means being responsible for your actions and decisions, and owning up to their outcomes.",
  "Ownership in work involves taking full responsibility for both successes and failures, without shifting blame.",
  "When you take accountability, you demonstrate integrity and trustworthiness, gaining respect from others.",
  "Owning your work means ensuring the quality and consistency of the results, even when challenges arise.",
  "Accountability builds a sense of responsibility, making you more reliable and dependable in your role.",
  "Being accountable helps improve performance because you are directly invested in the results.",
  "When you own your mistakes, it shows humility and a willingness to learn from past experiences.",
  "Accountability isn't just about acknowledging errors; it’s about taking proactive steps to improve.",
  "Ownership means embracing the process, not just the end result, and being involved at every stage.",
  "Being accountable gives you control over your work, allowing you to take initiative and drive change.",
  "Accountability drives self-improvement because you evaluate your performance critically and constructively.",
  "Taking ownership fosters a deeper connection to your role, increasing job satisfaction and personal growth.",
  "When everyone takes responsibility, the team works more effectively, as there’s clarity in roles and expectations.",
  "Accountability ensures that commitments are met on time, increasing efficiency and trust within a team.",
  "By owning your actions, you demonstrate leadership, regardless of your position in the organization.",
  "Accountability means communicating openly when there are issues, preventing misunderstandings and delays.",
  "When you take ownership, you become more engaged, which leads to better problem-solving and decision-making.",
  "Accountability is the foundation of strong teams, where everyone knows they can rely on one another.",
  "Owning up to your work results in personal empowerment and greater confidence in your decisions.",
  "Accountability also means setting an example for others, encouraging them to take responsibility as well.",
"Love having you with us! We appreciate all the hard work and your kind leadership in leading everyone of us here at WORQ. ",
"Thank you for keeping the worq space running so well! I’m glad to have you part of the team! ",
"It’s incredible to see you consistently pushing the bar, Tharsha. We are so grateful to have someone like you in our team who is willing to go above and beyond to achieve such great success for the team. As I recalled at the beginning of my employment with WORQ, we weren't good together, like a stranger, me joining you as a team, I was thinking hard, how can I get in?? But now, you have opened up, and I must let you know that I LOVE IT, working with you and at times listen to you, day or night, I enjoyed every single moment of it. Last but not least, Thank you for bringing your best to work every single day!",
"Your effort and time both mean a lot to all of us,Thank you for being an amazing manager .",
"Dear Ruhaya, your hard work and trustworthiness as our HRBP are truly appreciated, which I think will make a lasting impact at WORQ and we're fortunate to have you. Thank you for your valuable contributions and hope that you can achieve higher heights!",
"Dear Tiff, hope to thank you for your efforts that go beyond the workplace, and it's truly commendable. Thank you for being a compassionate HR manager who cares about our community's well-being. and also thank you for helping out on the fundraising engagements on numerous occasions!",
"We all know that the transition from being an Intern to a Permanent team members is not easy because the expectation is 100% higher BUT you always managed to handle it professionally. Thank you for being part of the team and spreading your positivity around us. You excel at handling tasks with minimal guidance and always ready to RUN the show has helped the team a lot. We could not thank you enough :) You're making great headway. GOOD JOB, IQYN!",
       
       
        "Your customer-centric approach is truly inspiring. You always go above and beyond to ensure that each customer's needs are met with exceptional care and attention.",
        "I’m always amazed by your ability to focus on the customer experience. Your passion for creating solutions that truly benefit our clients sets you apart.",
        "Thank you for always putting our customers first. Your unwavering dedication to their success speaks volumes about your commitment to our mission.",
        "Your customer-obsessed mindset is what drives our success. You always listen to feedback, anticipate needs, and create solutions that matter.",
        "Your ability to build lasting relationships with clients is incredible. You always ensure they feel heard and valued, and that makes a huge difference.",
        "I appreciate how you always think about the customer's journey. You consider every touchpoint and make sure it’s a positive experience for them.",
        "Thank you for your relentless pursuit of excellence in customer satisfaction. Your attention to detail and proactive problem-solving make a real impact.",
        "Your ability to understand and empathize with customers is something I deeply admire. You make them feel like a priority and always find a way to exceed their expectations.",
        "I am continually impressed by how you turn challenges into opportunities for our customers. Your innovative approach has made a significant difference to their experience.",
        "Your customer-first mentality is a great example for everyone. You consistently deliver value and go out of your way to ensure clients are happy and engaged.",
        "I admire how you bring a sense of community into everything you do. Whether working with colleagues or clients, you always prioritize meaningful relationships.",
        "Thank you for your passion for community. You’re constantly thinking about how to give back, whether through outreach or fostering a culture of support within the company.",
        "Your focus on building a supportive and inclusive environment for both customers and team members has created a thriving community culture that everyone appreciates.",
        "I appreciate your community-driven approach. You always look for ways to make a positive impact, whether it’s connecting with others or driving initiatives that benefit everyone.",
        "You consistently show a commitment to the community. Your willingness to collaborate, share knowledge, and foster inclusiveness makes a huge difference.",
        "Thank you for being such a positive force within our community. Your generosity, openness, and willingness to help others are truly inspiring.",
        "Your efforts in fostering a sense of belonging and community among our customers are unparalleled. You create an environment where everyone feels valued.",
        "I admire your dedication to not only serving customers but also strengthening our community. You always find ways to bring people together and create a supportive atmosphere.",
        "Your community-minded attitude has really set a strong foundation for both personal and professional growth. You care about the bigger picture, and it shows in everything you do.",
        "Thank you for your tireless dedication to both our customers and the community. Your willingness to help others and support meaningful causes enriches all of us.",
  "Being customer-obsessed means always prioritizing the needs and satisfaction of your customers.",
  "A customer-obsessed mindset involves listening attentively to feedback and continuously improving your services or products.",
  "Community-minded individuals focus on building strong, positive relationships within their local or online communities.",
  "Customer obsession goes beyond satisfaction; it's about creating exceptional experiences that exceed expectations.",
  "Being community-minded means giving back and contributing to the well-being of others around you.",
  "Customer obsession is not just about solving problems but anticipating needs before they arise.",
  "A community-minded mindset encourages collaboration, unity, and support among diverse groups of people.",
  "Customer-obsessed companies continuously innovate to deliver more value, creating loyal and long-lasting relationships.",
  "Being community-minded helps foster a culture of belonging and inclusivity for everyone involved.",
  "Customer obsession requires constant learning, adapting to changing customer needs and preferences.",
  "Community-minded individuals are often driven by empathy and a desire to make a positive impact on others.",
  "A customer-obsessed approach leads to deeper engagement and greater customer loyalty over time.",
  "Being community-minded involves supporting others and working together for collective success, not individual gain.",
  "Customer obsession means considering the customer's perspective in every decision and action taken by the company.",
  "Community-minded businesses focus on sustainability, ethics, and creating long-term positive effects for everyone involved.",
  "Customer-obsessed individuals proactively seek out and address pain points, making life easier for customers.",
  "Being community-minded involves looking out for the well-being of others and helping those in need whenever possible.",
  "Customer obsession is about creating not just a product or service, but a relationship built on trust and respect.",
  "A community-minded person promotes shared values and creates spaces where people can thrive together.",
  "Customer-obsessed companies focus on delivering results that are meaningful, making their customers' lives better.",
  "It’s an incredible work experience to work under a boss whose skills and talents are notable. I get to learn something new from you every day. Thank you for showing me how a great leader leads madam Majorie! ",
"Thanks for being such a great help! Hope you'll progress and received a lot of new skills from this internship!",
"Thank you for being such a good teammate . I am impressed by the way you welcoming our clients/members during we brought them for tours . Thank you .",
"I appreciate your hard work and dedication. Members at TTDI keep saying your good name regards your assistances.Good One !",
"Did not have a chance to say a proper thank you to you. Thank you for your guidance and patience during my internship. You've been so nice and patient at teaching me a lot of things, I hope it's not to late to say this. xoxo ",
"Thank you for helping me A LOT during Tiffany's not around. You hold the fort and sacrifice a lot in HR. I appreciate everything you've done for us. Always be happy and know that we noticed your hardworq. ",
"Thank you for taking up the QARMA Points revamp project and allow the team members to recognise each others. Thank you for always being patience and understanding on helping us in scaling the system. Your ownership on this project is highly appreciated :)",
"Thank you for helping us a lot in HR especially running the TikTok to promote our brand to more people! Learn and do as much mistakes during your internship, identify and improve yourself. We all learning :3",
"I appreciate your dedication to the team throughout these past few months. Your responsible approach sets a high standard, and your guidance to newer members is greatly appreciated. ",
"Thank you Majorie for being with us on this leadership journey",
       
        
        "Your entrepreneurial mindset is truly inspiring. You always approach challenges as opportunities, and your ability to innovate is a huge asset to the team.",
        "I admire your startup mentality. You have an incredible ability to take risks and adapt quickly, which helps drive the business forward.",
        "Your passion for growth and innovation is contagious. You consistently think outside the box and find creative solutions to complex problems.",
        "Thank you for always embracing the startup mentality. You see every setback as a learning opportunity, and that resilience helps propel the company to new heights.",
        "I appreciate how you constantly think like an entrepreneur, finding ways to improve processes and make bold decisions that move the needle for the business.",
        "Your ability to take initiative and move quickly is one of the things I admire most. You approach every task with a sense of urgency and ownership that drives results.",
        "Your entrepreneurial spirit is evident in everything you do. You’re always looking for ways to streamline, innovate, and create something that makes a difference.",
        "Thank you for your relentless drive to succeed. Your ability to focus on growth and build something meaningful out of challenges is truly inspiring.",
        "You embody the startup mindset every day. Your ability to pivot quickly and remain flexible has been crucial in keeping us on track to reach our goals.",
        "Your innovative thinking and ability to create value out of uncertainty are exactly what make you a true entrepreneur. You continuously inspire the team to aim higher.",
        "I’m always amazed at your willingness to take calculated risks. Your entrepreneurial mindset allows you to embrace uncertainty with confidence and lead by example.",
        "Thank you for bringing a fresh, entrepreneurial perspective to everything you do. Your ideas and solutions are always forward-thinking and ready for scaling.",
        "Your ability to tackle challenges with a startup mentality has been pivotal in moving the company forward. You don't shy away from tough decisions, and it’s inspiring.",
        "I admire your vision and leadership. You think beyond the present, always looking for ways to future-proof the business and stay ahead of the curve.",
        "Thank you for your determination and passion for innovation. You constantly push the boundaries, and your startup mindset is a key driver of our success.",
        "Your entrepreneurial mindset is an asset to the team. You think strategically and focus on creating long-term value, helping to build the foundation for future success.",
        "I love your startup mentality. You're always ready to iterate, pivot, and learn, and that mindset is exactly what we need to grow and succeed.",
        "Your entrepreneurial mindset has been pivotal in bringing fresh perspectives and solutions to the table. You approach each challenge with excitement and energy.",
        "Thank you for constantly challenging the status quo and looking for innovative ways to improve. Your entrepreneurial drive is what makes you stand out and inspires others.",
        "Your ability to operate in ambiguity and take ownership of new projects is exactly what makes you a true entrepreneur. You make things happen and inspire everyone around you.",
     "The startup mindset involves embracing uncertainty and viewing challenges as opportunities for growth.",
  "Entrepreneurs are driven by a vision to solve problems and create value, no matter the obstacles they face.",
  "Startups thrive on innovation and creativity, constantly pushing boundaries to disrupt industries and markets.",
  "An entrepreneurial mindset is about resilience, adapting quickly to change and learning from failure.",
  "Startup founders embrace risk as an essential part of the journey, understanding that growth requires taking chances.",
  "Entrepreneurs focus on execution, turning ideas into action and driving results despite limited resources.",
  "In a startup environment, flexibility is key; entrepreneurs must pivot quickly when needed and adjust strategies.",
  "The entrepreneurial mindset encourages a deep passion for what you do, fueling determination to succeed against all odds.",
  "Startups operate with a sense of urgency, prioritizing speed and innovation to get ahead of competitors.",
  "Entrepreneurs must possess the ability to see potential where others see obstacles, constantly searching for new solutions.",
  "In a startup, every team member wears multiple hats and collaborates across roles to make the business succeed.",
  "Entrepreneurship is about being scrappy, making the most out of available resources and constantly optimizing processes.",
  "Startups are built on a foundation of relentless problem-solving and a drive to bring something new to the market.",
  "An entrepreneurial mindset fosters a culture of ownership, where individuals take responsibility for their work and its outcomes.",
  "Entrepreneurs are always looking for ways to scale, seeking efficiencies that allow them to grow faster with fewer resources.",
  "Startups thrive on the ability to move fast, adapt to market changes, and iterate quickly based on customer feedback.",
  "Entrepreneurs embrace failure as part of the learning process, understanding that each setback brings new lessons.",
  "A startup mindset involves questioning the status quo, challenging traditional ways of doing things, and seeking innovative solutions.",
  "Entrepreneurs understand that success is built on perseverance and the willingness to keep going, even when things are tough.",
  "The entrepreneurial mindset values creativity and constant reinvention, knowing that staying stagnant can lead to failure."

   
    ],
    "label": [  # Labels: 1 = Customer Obsessed, 0 = Accountability & Ownership, 2 = Entrepreneurship
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0,0, 0, 0, 0,
  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ],
}


print(len(data["text"]))
print(len(data["label"]))

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())

# Convert text into numeric form using TF-IDF
# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Convert the text into numeric form using TF-IDF
X = vectorizer.fit_transform(df["text"])

# Labels
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
