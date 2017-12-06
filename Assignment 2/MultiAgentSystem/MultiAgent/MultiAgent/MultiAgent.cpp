// MultiAgent.cpp : Defines the entry point for the console application.
//
#include <vector>
#include <algorithm>
#include <random>
#include "time.h"
#include <iomanip>
#include <iostream>	

class Agent
{
public:
	Agent(bool agent_id, float credits, int current_state);	// current_state = 0
	~Agent();

	// Functions

	void updateCredits(double credit);					// Done after auction, trade, penalty and reward
	double getCredit();									// Return an agent's credits

														// Utility and Auction functions
	void computeUtility(double price);					// Calculates utilitiy values for the agent, refreshed several times
	int decideState(double price);						// Update current state (Do an action)
	double energyNeededFromC(int state);

	// Trade
	bool trade(Agent* a);								// Initiate trade with agent
	void resetRMatrix();								// Restore R matrix back to original values after auction


														// Energy functions
	void updateTemp(int indoor, int outdoor);			// Set's new values for temperatures if they change
	float computeEnergy(int indoortemp);				// Calculate needed energy volume based on indoor temp and outdoor temp
	void updateEnergy(int outdoortemp);

	int cs;
	std::vector< std::vector<int> > R_matrix;			// Reward matrix, public for trading
	std::vector<int> temperature;
	std::vector<float> energyCostVector;
	std::vector<double> U_values;

	int atBoundaries();									// Check if current_state is at the starting- or ending points of the Utility vector

private:
	// Variables
	bool agent_id;

	double credit;										// Updated after every auction
	float indoortemp, outdoortemp;
	double probabilities[3] = { 0.5, 0.2, 0.3 };		// Probability for moving a certain way; 0 = Stay, 1 = Up, 2 = Down
	double discount_factor = 0.7;

	double priceFromC;
	std::vector< std::vector<int> > original_reward;	// Used to restore R matrix after auctions in case a trade has taken place

};
Agent::Agent(bool id, float credits, int current_state)
{
	this->agent_id = id;
	this->credit = credits;
	this->cs = current_state;

	// Agent A
	if (id == true) {
		//						Stay, Up, Down
		std::vector<int> v0 = { -10, -5, 9999 };
		std::vector<int> v1 = { -5, 4, -10 };
		std::vector<int> v2 = { 1, 5, -5 };
		std::vector<int> v3 = { 5, 6, 4 };
		std::vector<int> v4 = { 6, -5, 5 };
		std::vector<int> v5 = { -5, -10, 10 };
		std::vector<int> v6 = { -10, 9999, -5 };

		R_matrix = { v0, v1, v2, v3, v4, v5, v6 };
		original_reward = R_matrix;						// Used for resetting
		U_values = { 0, 0, 0, 0, 0, 0, 0 };
		temperature = { 21,22,23,24,25,26,27 };

		this->indoortemp = 0;
		computeEnergy(indoortemp);
	}
	// Agent B
	else {
		//						Stay, Up, Down
		std::vector<int> v0 = { -10, -5, 9999 };
		std::vector<int> v1 = { -5, 2, -10 };
		std::vector<int> v2 = { 2, 4, -5 };
		std::vector<int> v3 = { 4, 5, 2 };
		std::vector<int> v4 = { 5, 6, 4 };
		std::vector<int> v5 = { 6, -5, 5 };
		std::vector<int> v6 = { -5, -10, 10 };
		std::vector<int> v7 = { -10, 9999, -5 };

		R_matrix = { v0, v1, v2, v3, v4, v5, v6, v7 };
		original_reward = R_matrix;						// Used for resetting
		U_values = { 0, 0, 0, 0, 0, 0, 0, 0 };
		temperature = { 18,19,20,21,22,23,24,25 };

		this->indoortemp = 0;
		computeEnergy(indoortemp);

	}

}

Agent::~Agent()
{
}

void Agent::updateCredits(double c)
{
	this->credit += c;
}

double Agent::getCredit() {

	return this->credit;
}

void Agent::resetRMatrix() {

	this->R_matrix = original_reward;
}

void Agent::computeUtility(double price)
{
	// Vectors for storing current and future utility values
	std::vector<double> v;
	std::vector<double> u;
	if (agent_id == true) v = { 0, 0, 0, 0, 0, 0, 0 };
	else v = { 0, 0, 0, 0, 0, 0, 0, 0 };
	u = v;

	// Check
	auto delta = 0;
	auto epsilon = 1e-3;

	// Update utility values until equilibrium
	do {
		u = v;
		delta = 0;

		for (unsigned int i = 0; i < U_values.size(); i++) {

			double cost = computeEnergy(temperature[i])* price * 6;
			if (i == 0) v[i] = (R_matrix[i][0] + discount_factor * std::max(probabilities[0] * u[i], probabilities[1] * u[i + 1])) - cost;
			else if (i == U_values.size() - 1) 	v[i] = R_matrix[i][0] + discount_factor * std::max(probabilities[0] * u[i], probabilities[2] * u[i - 1]) - cost;
			else v[i] = R_matrix[i][0] + discount_factor * std::max(std::max(probabilities[0] * u[i], probabilities[1] * u[i + 1]), probabilities[2] * u[i - 1]) - cost;


			if (std::abs(v[i] - u[i]) > delta) delta = std::abs(v[i] - u[i]);	// Check for equilibrium
		}

	} while ((delta > epsilon * (1 - discount_factor) / discount_factor));

	U_values = u;
}

bool Agent::trade(Agent* a) {

	// Check to see if current_state is at one of the ending points of the utility vector
	auto check = atBoundaries();	// 0 = both ok, 1 down is possible, 2 up is possible

	if (check == 0 || check == 1)	// If you are already at the lowest temperature, you can't go down. You are basically dead, good luck in Hades.
	{
		float credit = this->credit;
		float previous_utility = this->U_values[cs - 1];
		float current_utility = this->U_values[cs];
		float loss = current_utility - previous_utility;

		if (loss > (credit*0.10)) {	// Will experience great loss, therefore initiate trade

			float reward = R_matrix[cs][2];
			float price = std::abs(reward) * 0.8;
			a->R_matrix[a->cs][2] += price;
			return true;
		}

		else return false;
	}
}

double Agent::energyNeededFromC(int state)
{
	return energyCostVector[state];
}

void Agent::updateTemp(int indoor, int outdoor)
{
	indoortemp = indoor;
	outdoortemp = outdoor;
}


//Decide what the next state will be based on utility vector
int Agent::decideState(double price) {

	priceFromC = price;
	computeUtility(price);
	double state;				// return state

	double stay = U_values[cs];
	double down = -999;
	double up = -999;

	// Check to see if current_state is at one of the ending points of the utility vector
	auto check = atBoundaries();	// 0 = both ok, 1 down is possible, 2 up is possible

									// If there is a state below current state, get it's utility
	if (check == 0 || check == 1) down = U_values[cs - 1];

	// If there is a state above current state, get it's utility
	if (check == 0 || check == 2) up = U_values[cs + 1];

	// Find the state with the most utility
	state = std::max(stay, std::max(down, up));
	int i = 0;
	for (; i <= U_values.size(); ++i) {

		//std::cout<<"i= "<<i<<" size" << U_values.size();
		//if (U_values[i] == state) break;
	}

	return temperature[i];

}

// Change in main whenever outTemp is changeing
void Agent::updateEnergy(int out) {

	for (int i = 0; i < temperature.size(); i++) {
		// Agent A
		if (agent_id == true) {

			energyCostVector.push_back(0.2 + 0.5 * (temperature[i] - out));	// Energy required to maintain temp for 1 hour
		}
		// Agent B
		else {
			energyCostVector.push_back(0.25 + 0.5 * (temperature[i] - out));	// Energy required to maintain temp for 1 hour
		}
	}
}

float Agent::computeEnergy(int indoortemp)
{
	float energy = 0;

	// Agent A
	if (agent_id == true) {

		energy = 0.2 + 0.5 * (indoortemp - outdoortemp);	// Energy required to maintain temp for 1 hour
	}
	// Agent B
	else {
		energy = 0.25 + 0.5 * (indoortemp - outdoortemp);	// Energy required to maintain temp for 1 hour
	}

	return energy;
}

int Agent::atBoundaries() {

	std::vector<double>::iterator it;
	int check = 0;
	for (it = this->U_values.begin(); it <= U_values.end(); it++) {

		if (*it == U_values[cs]) break;
		else if (it == U_values.end()) check = 999;
		else check += 1;
	}

	if (check != 0 && check != 999) return 0;
	else {
		if (check != 0) return 1;
		if (check != 999) return 2;
	}
}

//Auctioner
class Auctioner {
public:
	Auctioner(int credit, float start_limit);
	~Auctioner();

	void updateCeiling(double energy);		// Update maximum available energy
	void runAuction();						// Start auction
	double assessPrice();					// Look at ceiling and 
	void updateCredits(double credit);
	double getCredit();

	int assessLoad(double a, double b);		// Check if the sum of bids from A and B are on, above or below the ceiling
	bool active_auction = false;


private:
	double credit;
	double ceiling;
	double limit = 4.5;
};

Auctioner::Auctioner(int credit, float start_limit)
{
	this->credit = credit;
	this->ceiling = start_limit;		// Should be 4.5;
	srand(time(NULL));
}

Auctioner::~Auctioner()
{
}

void Auctioner::updateCredits(double credit)
{
	this->credit += credit;
}

double Auctioner::getCredit()
{
	return this->credit;
}

void Auctioner::updateCeiling(double energy)
{
	auto r = ((double)rand() / (RAND_MAX));
	//auto r = rand() % 2; // random number between 0 and one
	limit += r * 5;

	ceiling = (limit + energy) * 6;		// Energy cost for both agens were a lot higher than the ceiling, so we are raising the ceiling to compensate
}

void Auctioner::runAuction()
{
	this->active_auction = true;
}

double Auctioner::assessPrice() {

	if (ceiling > 45) return 0.6;	// If the ceiling is high, C tries to get a high price for the energy
	else return 0.3;				// If the ceiling is not high, C uses the standard price for energy in Norway
}

int Auctioner::assessLoad(double a, double b) {

	a *= 6;
	b *= 6;

	if ((a + b) < ceiling - 0.5) return 1;
	else if ((a + b) > ceiling) return 2;
	else return 0;
}



//Solar Panel
class SolarPanel {

public:
	SolarPanel() {};
	~SolarPanel() {};

	float energy();					// Return energy to agent C
	void updateWeather();			// Update the current weather
	void set_weather(int w);		// Set the current weather

private:

	int peak_value = 3;
	float s;
	std::vector<float> clouds = { 1.0f, 0.7f, 0.5f, 0.3f, 0.1f };
	int weather = 3;


};

float SolarPanel::energy()
{
	float energy = peak_value * s;
	return energy;
}

void SolarPanel::updateWeather()
{
	s = clouds[weather];
}

void SolarPanel::set_weather(int w) //pass 0,1,2,3,4
{
	weather = w;
	updateWeather();
}

void fill_outside_temperatures(std::vector<int> *v) {

	// From 06 to 18;	From 18 to 06
	v->push_back(2);	v->push_back(3);
	v->push_back(2);	v->push_back(3);
	v->push_back(2);	v->push_back(3);
	v->push_back(2);	v->push_back(3);
	v->push_back(2);	v->push_back(3);
	v->push_back(3);	v->push_back(3);
	v->push_back(3);	v->push_back(3);
	v->push_back(3);	v->push_back(3);
	v->push_back(3);	v->push_back(3);
	v->push_back(2);	v->push_back(2);
	v->push_back(2);	v->push_back(2);
	v->push_back(2);	v->push_back(2);
	v->push_back(2);	v->push_back(2);
}

int find_agent_index(Agent* a, double s) {

	int i = 0;
	for (; i < a->temperature.size(); i++) {
		if (a->temperature[i] == s) break;
	}
	return i;
}

void announceWinner(Agent* a, Agent* b, int iterations) {

	char won;
	char lost;
	double winning_credit;
	double losing_credit;

	if (a->getCredit() < b->getCredit())
	{
		won = 'B';
		lost = 'A';
		winning_credit = b->getCredit();
		losing_credit = a->getCredit();
	}
	else {
		won = 'A';
		lost = 'B';
		winning_credit = a->getCredit();
		losing_credit = b->getCredit();
	}

	std::cout << "\nThe trade has ended after " << iterations << " iterations" << std::endl;
	std::cout << std::setprecision(9) << std::showpoint << std::fixed;
	std::cout << "Winner is Agent " << won << " with " << winning_credit << " credits!" << std::endl;
	std::cout << "Agent " << lost << " lost with " << losing_credit << " credits!" << std::endl;
	std::cout << "Enter q to exit"<< std::endl;
}


//Program Main()
int main(int argc, const char * argv[]) {
	
	do {
		Agent* a = new Agent(1, 500, 0);
		Agent* b = new Agent(0, 500, 0);
		Auctioner* c = new Auctioner(10000, 4.5);

		int iterations = 0;

		// Weather variables
		std::vector<int> outside_temperatures;
		fill_outside_temperatures(&outside_temperatures);

		// Energy variables
		SolarPanel* solar = new SolarPanel();
		solar->set_weather(2);	// Start at partly cloudy
		auto energy = solar->energy();
		c->updateCeiling(energy);

		a->updateEnergy(outside_temperatures[0]);
		b->updateEnergy(outside_temperatures[0]);

		auto hour_counter = 0;		// Too keep track of which 6 hour period of the full day we are currently in
		auto stop = false;

		// While the agents have at least 1 credit
		while (stop == false)
		{
			if (a->getCredit() < 0.0) stop = true;
			if (b->getCredit() < 0.0) stop = true;
			if (iterations >= 50) stop = true;



			int timestep = 1;
			for (; timestep <= 6; timestep++)
			{
				if (timestep == 6) {

					c->active_auction = true;
					auto price = c->assessPrice();

					// Find the temperature the agents want to be in based on the price
					auto state_a = a->decideState(price);
					auto state_b = b->decideState(price);

					// Will be used to connect energy cost (energyCostVector) to current temperature of the agent (state_a/state_b)
					// If agent A is at temp 22 then that means it is at index 2, if agent B is at temp 19 that means it is at index 2 and so on, which corresponds to the index of the energy costs
					auto index_a = find_agent_index(a, a->decideState(price));
					auto index_b = find_agent_index(b, a->decideState(price));

					double energy_a = a->energyNeededFromC(index_a);
					double energy_b = b->energyNeededFromC(index_b);

					while (c->active_auction == true)
					{
						auto energy_load = c->assessLoad(energy_a, energy_b);

						if (energy_load == 0)
						{
							c->active_auction = false;	// Pareto efficiency has been achieved. Auction is done
						}
						else if (energy_load == 1) {	// Below the ceiling, lower the price
							price = price * 0.6;
						}
						else if (energy_load == 2) {		// Sum higher than ceiling

															// Check if the agent wants to trade
							auto a_trade = a->trade(b);
							auto b_trade = b->trade(a);

							if (a_trade == false && b_trade == false) {		// No trade, increase price. Else, return to top and check again
								price = price * 1.6;
							}
						}
						auto old_energy_a = energy_a;
						auto old_energy_b = energy_b;

						// Need to find if the agents lowered or raised their temperature when given the starting price to accurately gain the next temperature
						auto temp_a = a->temperature[a->cs];
						auto temp_b = b->temperature[b->cs];

						if (temp_a < state_a) a->cs++;
						else if (temp_a > state_a) a->cs--;

						if (temp_b < state_b) b->cs++;
						else if (temp_b > state_b) b->cs--;

						state_a = a->decideState(price);
						state_b = b->decideState(price);

						index_a = find_agent_index(a, state_a);
						index_b = find_agent_index(b, state_b);

						energy_a = a->energyNeededFromC(index_a);
						energy_b = b->energyNeededFromC(index_b);

						if ((old_energy_a == energy_a && old_energy_b == energy_b) && (c->assessLoad(energy_a, energy_b == 1))) {	// Below the ceiling, but no change in demand

							c->active_auction = false;	// Auction is done
						}
					}
					// After the auction

					// Reset reward matrices in case trading has happened
					a->resetRMatrix();
					b->resetRMatrix();

					// Update state for A and B
					a->cs = index_a;
					b->cs = index_b;

					// Update Credits for A, B, C
					auto credits_a = a->U_values[index_a];
					auto credits_b = b->U_values[index_b];
					auto credits_c = (energy_b * price) * 6 + (energy_b * price) * 6;

					a->updateCredits(credits_a);
					b->updateCredits(credits_b);
					c->updateCredits(credits_c);

					iterations++;
				}

				// Do things that happen at every timestep, update ceiling and change the weather

				// Gain solar power only during the daytime
				if (hour_counter < 2) {
					auto random_weather = rand() % 5;
					solar->set_weather(random_weather);
					solar->updateWeather();
					energy = solar->energy();
					c->updateCeiling(energy);
				}

				// Update energy requirements based on new temperatures
				a->energyCostVector.clear();
				b->energyCostVector.clear();
				a->updateEnergy(outside_temperatures[timestep + hour_counter]);
				b->updateEnergy(outside_temperatures[timestep + hour_counter]);
			}

			if (hour_counter <= 3) hour_counter++;
			else hour_counter = 0;						// We have done a full day, and are starting over
		}

		announceWinner(a, b, iterations);
	}while (std::getchar() != 'q');
}