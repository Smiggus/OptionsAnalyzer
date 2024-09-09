import numpy as np
import scipy.stats as stats

""" Note: This code is for FX options, not equity options. """

class Option:
    def __init__(self, strike, expiry, option_type, implied_vol, spot, interest_rate):
        """
        Initialize an option.

        Args:
            strike (float): Strike price of the option.
            expiry (float): Time to expiry in years.
            option_type (str): Type of the option ('call' or 'put').
            implied_vol (float): Implied volatility of the option.
            spot (float): Current spot price of the underlying.
            interest_rate (float): Risk-free interest rate.
        """
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type
        self.implied_vol = implied_vol
        self.spot = spot
        self.interest_rate = interest_rate

    def d1_d2(self):
        """
        Helper function to calculate d1 and d2 for Black-Scholes formulas.
        
        Returns:
            tuple: (d1, d2) used for calculating the Greeks and option price.
        """
        d1 = (np.log(self.spot / self.strike) +
              (self.interest_rate + 0.5 * self.implied_vol ** 2) * self.expiry) / (self.implied_vol * np.sqrt(self.expiry))
        d2 = d1 - self.implied_vol * np.sqrt(self.expiry)
        return d1, d2

    def black_scholes(self):
        """
        Calculate the price of the option using the Black-Scholes model.
        
        Returns:
            float: Option price.
        """
        d1, d2 = self.d1_d2()

        if self.option_type == 'call':
            price = (self.spot * stats.norm.cdf(d1) -
                     self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2) -
                     self.spot * stats.norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        return price

    def delta(self):
        """
        Calculate the delta of the option.

        Returns:
            float: Option delta.
        """
        d1, _ = self.d1_d2()

        if self.option_type == 'call':
            return stats.norm.cdf(d1)
        elif self.option_type == 'put':
            return stats.norm.cdf(d1) - 1

    def gamma(self):
        """
        Calculate the gamma of the option.

        Returns:
            float: Option gamma.
        """
        d1, _ = self.d1_d2()
        gamma = stats.norm.pdf(d1) / (self.spot * self.implied_vol * np.sqrt(self.expiry))
        return gamma

    def vega(self):
        """
        Calculate the vega of the option (sensitivity to volatility).

        Returns:
            float: Option vega.
        """
        d1, _ = self.d1_d2()
        vega = self.spot * stats.norm.pdf(d1) * np.sqrt(self.expiry)
        return vega / 100  # Typically expressed in percentage points.

    def theta(self):
        """
        Calculate the theta of the option (time decay).

        Returns:
            float: Option theta.
        """
        d1, d2 = self.d1_d2()

        if self.option_type == 'call':
            theta = (- (self.spot * stats.norm.pdf(d1) * self.implied_vol) / (2 * np.sqrt(self.expiry)) -
                     self.interest_rate * self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2))
        elif self.option_type == 'put':
            theta = (- (self.spot * stats.norm.pdf(d1) * self.implied_vol) / (2 * np.sqrt(self.expiry)) +
                     self.interest_rate * self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2))

        return theta / 365  # Typically expressed per day.

    def rho(self):
        """
        Calculate the rho of the option (sensitivity to interest rate changes).

        Returns:
            float: Option rho.
        """
        _, d2 = self.d1_d2()

        if self.option_type == 'call':
            rho = self.strike * self.expiry * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2)
        elif self.option_type == 'put':
            rho = -self.strike * self.expiry * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2)

        return rho / 100  # Typically expressed in percentage points.


class Strategy:
    def __init__(self, options):
        """
        Base class for options strategies.

        Args:
            options (list): List of Option objects.
        """
        self.options = options

    def strategy_price(self):
        """
        Calculate the total price of the strategy.

        Returns:
            float: Total price of all options in the strategy.
        """
        total_price = sum(option.black_scholes() for option in self.options)
        return total_price

    def strategy_delta(self):
        """
        Calculate the total delta of the strategy.

        Returns:
            float: Total delta of all options in the strategy.
        """
        total_delta = sum(option.delta() for option in self.options)
        return total_delta


class RiskReversal(Strategy):
    def __init__(self, put, call):
        """
        Initialize a risk reversal strategy.

        Args:
            put (Option): A put option.
            call (Option): A call option.
        """
        super().__init__([put, call])


class Straddle(Strategy):
    def __init__(self, put, call):
        """
        Initialize a straddle strategy.

        Args:
            put (Option): A put option.
            call (Option): A call option.
        """
        super().__init__([put, call])


class Butterfly(Strategy):
    def __init__(self, lower_strike, middle_strike, upper_strike):
        """
        Initialize a butterfly spread strategy.

        Args:
            lower_strike (Option): An option at a lower strike.
            middle_strike (Option): Two options at a middle strike.
            upper_strike (Option): An option at a higher strike.
        """
        super().__init__([lower_strike, middle_strike, middle_strike, upper_strike])


class VolatilityAnalyzer:
    @staticmethod
    def z_score(implied_vols, strike):
        """
        Calculate the Z-score for a given strike's implied volatility.

        Args:
            implied_vols (dict): Dictionary of implied volatilities with strike prices as keys.
            strike (float): The strike price for which to calculate the Z-score.

        Returns:
            float: Z-score for the implied volatility at the given strike.
        """
        # Extract the implied volatilities as a list
        vol_values = list(implied_vols.values())
        
        mean_vol = np.mean(vol_values)
        std_vol = np.std(vol_values)
        
        # Get the implied volatility for the given strike
        strike_vol = implied_vols.get(strike)

        if strike_vol is None:
            raise ValueError(f"No implied volatility found for strike {strike}")

        z_score = (strike_vol - mean_vol) / std_vol
        return z_score

    @staticmethod
    def break_even(strategy, underlying_move):
        """
        Calculate the break-even point of the strategy.

        Args:
            strategy (Strategy): The options strategy being analyzed.
            underlying_move (float): Expected move in the underlying asset.

        Returns:
            float: The break-even price of the strategy.
        """
        return strategy.strategy_price() / underlying_move


def main():
    # Example usage of the program
    spot_price = 145
    interest_rate = 0.01
    expiry = 0.5  # 6 months
    implied_vols = {140: 0.15, 145: 0.12, 150: 0.10}

    # Create individual options
    put_option = Option(strike=143, expiry=expiry, option_type='put',
                        implied_vol=implied_vols[140], spot=spot_price, interest_rate=interest_rate)
    call_option = Option(strike=147, expiry=expiry, option_type='call',
                         implied_vol=implied_vols[150], spot=spot_price, interest_rate=interest_rate)

    # Example: Risk Reversal strategy
    risk_reversal = RiskReversal(put=put_option, call=call_option)
    print(f"Risk Reversal Strategy Price: {risk_reversal.strategy_price()}")
    print(f"Risk Reversal Strategy Delta: {risk_reversal.strategy_delta()}")

    # Print Greeks for individual options
    print("\n--- Put Option Greeks ---")
    print(f"Delta: {put_option.delta()}")
    print(f"Gamma: {put_option.gamma()}")
    print(f"Vega: {put_option.vega()}")
    print(f"Theta: {put_option.theta()}")
    print(f"Rho: {put_option.rho()}")

    print("\n--- Call Option Greeks ---")
    print(f"Delta: {call_option.delta()}")
    print(f"Gamma: {call_option.gamma()}")
    print(f"Vega: {call_option.vega()}")
    print(f"Theta: {call_option.theta()}")
    print(f"Rho: {call_option.rho()}")

    # Z-score calculation for strike 145
    volatility_analyzer = VolatilityAnalyzer()
    z_score = volatility_analyzer.z_score(implied_vols, strike=145)
    print(f"\nZ-Score for Strike 145: {z_score}")


if __name__ == "__main__":
    main()
