import numpy as np
import scipy.stats as stats


class EquityOption:
    def __init__(self, strike, expiry, option_type, implied_vol, spot, interest_rate, dividend_yield):
        """
        Initialize an equity option with a dividend yield.

        Args:
            strike (float): Strike price of the option.
            expiry (float): Time to expiry in years.
            option_type (str): Type of the option ('call' or 'put').
            implied_vol (float): Implied volatility of the option.
            spot (float): Current spot price of the underlying.
            interest_rate (float): Risk-free interest rate.
            dividend_yield (float): Continuous dividend yield.
        """
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type
        self.implied_vol = implied_vol
        self.spot = spot
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield

    def d1_d2(self):
        """
        Helper function to calculate d1 and d2 for the Black-Scholes-Merton formula.

        Returns:
            tuple: (d1, d2) used for calculating the Greeks and option price.
        """
        d1 = (np.log(self.spot / self.strike) +
              (self.interest_rate - self.dividend_yield + 0.5 * self.implied_vol ** 2) * self.expiry) / (self.implied_vol * np.sqrt(self.expiry))
        d2 = d1 - self.implied_vol * np.sqrt(self.expiry)
        return d1, d2

    def black_scholes_merton(self):
        """
        Calculate the price of the option using the Black-Scholes-Merton model for equity options.

        Returns:
            float: Option price.
        """
        d1, d2 = self.d1_d2()

        if self.option_type == 'call':
            price = (self.spot * np.exp(-self.dividend_yield * self.expiry) * stats.norm.cdf(d1) -
                     self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2) -
                     self.spot * np.exp(-self.dividend_yield * self.expiry) * stats.norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        return price

    def delta(self):
        """
        Calculate the delta of the equity option.

        Returns:
            float: Option delta.
        """
        d1, _ = self.d1_d2()

        if self.option_type == 'call':
            return np.exp(-self.dividend_yield * self.expiry) * stats.norm.cdf(d1)
        elif self.option_type == 'put':
            return np.exp(-self.dividend_yield * self.expiry) * (stats.norm.cdf(d1) - 1)

    def gamma(self):
        """
        Calculate the gamma of the equity option.

        Returns:
            float: Option gamma.
        """
        d1, _ = self.d1_d2()
        gamma = stats.norm.pdf(d1) / (self.spot * self.implied_vol * np.sqrt(self.expiry))
        return gamma

    def vega(self):
        """
        Calculate the vega of the equity option (sensitivity to volatility).

        Returns:
            float: Option vega.
        """
        d1, _ = self.d1_d2()
        vega = self.spot * np.exp(-self.dividend_yield * self.expiry) * stats.norm.pdf(d1) * np.sqrt(self.expiry)
        return vega / 100  # Typically expressed in percentage points.

    def theta(self):
        """
        Calculate the theta of the equity option (time decay).

        Returns:
            float: Option theta.
        """
        d1, d2 = self.d1_d2()

        if self.option_type == 'call':
            theta = (- (self.spot * stats.norm.pdf(d1) * self.implied_vol * np.exp(-self.dividend_yield * self.expiry)) / (2 * np.sqrt(self.expiry))
                     - self.interest_rate * self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2)
                     + self.dividend_yield * self.spot * np.exp(-self.dividend_yield * self.expiry) * stats.norm.cdf(d1))
        elif self.option_type == 'put':
            theta = (- (self.spot * stats.norm.pdf(d1) * self.implied_vol * np.exp(-self.dividend_yield * self.expiry)) / (2 * np.sqrt(self.expiry))
                     + self.interest_rate * self.strike * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2)
                     - self.dividend_yield * self.spot * np.exp(-self.dividend_yield * self.expiry) * stats.norm.cdf(-d1))

        return theta / 365  # Typically expressed per day.

    def rho(self):
        """
        Calculate the rho of the equity option (sensitivity to interest rate changes).

        Returns:
            float: Option rho.
        """
        _, d2 = self.d1_d2()

        if self.option_type == 'call':
            rho = self.strike * self.expiry * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(d2)
        elif self.option_type == 'put':
            rho = -self.strike * self.expiry * np.exp(-self.interest_rate * self.expiry) * stats.norm.cdf(-d2)

        return rho / 100  # Typically expressed in percentage points.


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


def main():
    # Example usage of the program for equity options
    spot_price = 100
    interest_rate = 0.02  # 2% risk-free interest rate
    dividend_yield = 0.03  # 3% continuous dividend yield
    expiry = 1  # 1 year until expiration
    implied_vols = {95: 0.2, 100: 0.18, 105: 0.16}

    # Create individual options
    put_option = EquityOption(strike=95, expiry=expiry, option_type='put',
                              implied_vol=implied_vols[95], spot=spot_price, interest_rate=interest_rate, dividend_yield=dividend_yield)
    call_option = EquityOption(strike=105, expiry=expiry, option_type='call',
                               implied_vol=implied_vols[105], spot=spot_price, interest_rate=interest_rate, dividend_yield=dividend_yield)

    # Display Greeks and option price for both options
    print("\n--- Put Option ---")
    print(f"Price: {put_option.black_scholes_merton()}")
    print(f"Delta: {put_option.delta()}")
    print(f"Gamma: {put_option.gamma()}")
    print(f"Vega: {put_option.vega()}")
    print(f"Theta: {put_option.theta()}")
    print(f"Rho: {put_option.rho()}")

    print("\n--- Call Option ---")
    print(f"Price: {call_option.black_scholes_merton()}")
    print(f"Delta: {call_option.delta()}")
    print(f"Gamma: {call_option.gamma()}")
    print(f"Vega: {call_option.vega()}")
    print(f"Theta: {call_option.theta()}")
    print(f"Rho: {call_option.rho()}")

    # Z-score calculation for strike 100
    volatility_analyzer = VolatilityAnalyzer()
    z_score = volatility_analyzer.z_score(implied_vols, strike=100)
    print(f"\nZ-Score for Strike 100: {z_score}")


if __name__ == "__main__":
    main()
