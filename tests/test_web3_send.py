"""Tests for the web3 BNB auto-deposit helpers.

Covers:
  1. _EVM_ADDR_RE rejects TRX-style addresses, BTC, garbage; accepts 0x...40hex.
  2. send_bnb_deposit() refuses to send to an invalid address.
  3. send_bnb_deposit() refuses to run when BSC_PRIVATE_KEY is missing.
  4. send_bnb_deposit() retries 3x on RPC failure, then raises (no real send).

We DO NOT broadcast a real transaction here - retry tests mock RPC. The only
network call is a chain-id sanity probe, which we monkey-patch to a fake w3.
"""
import os
import sys
import unittest
from decimal import Decimal
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import styx_register  # noqa: E402


GOOD_ADDR = "0x1877131C4001dDF19D37E73F18b09dF6a35757e6"


class TestEvmAddrRegex(unittest.TestCase):
    def test_accepts_real_evm(self):
        self.assertTrue(styx_register._EVM_ADDR_RE.fullmatch(GOOD_ADDR))

    def test_rejects_trx(self):
        # TRX uses base58 starting with 'T', length 34
        self.assertIsNone(styx_register._EVM_ADDR_RE.fullmatch(
            "TXyZ1234567890123456789012345678901"))

    def test_rejects_short(self):
        self.assertIsNone(styx_register._EVM_ADDR_RE.fullmatch("0x1234"))

    def test_rejects_no_prefix(self):
        self.assertIsNone(styx_register._EVM_ADDR_RE.fullmatch(
            "1877131C4001dDF19D37E73F18b09dF6a35757e6"))


class TestSendBnbDeposit(unittest.TestCase):
    def setUp(self):
        # Snapshot env so each test can mutate freely
        self._env_snapshot = {
            k: os.environ.get(k) for k in
            ("BSC_PRIVATE_KEY", "BSC_RPC_URL", "BSC_RPC_URL_FALLBACK")
        }

    def tearDown(self):
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_invalid_address_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            styx_register.send_bnb_deposit(
                to_address="not-an-evm-address", amount_bnb=0.001,
                max_retries=1)
        self.assertIn("invalid EVM address", str(ctx.exception))

    def test_missing_private_key_raises(self):
        os.environ.pop("BSC_PRIVATE_KEY", None)
        with self.assertRaises(RuntimeError) as ctx:
            styx_register.send_bnb_deposit(
                to_address=GOOD_ADDR, amount_bnb=0.001, max_retries=1)
        self.assertIn("BSC_PRIVATE_KEY", str(ctx.exception))

    def test_retries_three_times_then_raises(self):
        # Ensure key is set so we get past the env preflight. We generate
        # a fresh random throwaway 32-byte key per test run so no real
        # private key is ever hardcoded in source / git history.
        import secrets
        os.environ["BSC_PRIVATE_KEY"] = "0x" + secrets.token_hex(32)

        # Build a fake w3 object whose send_raw_transaction always blows up.
        fake_w3 = mock.MagicMock()
        fake_w3.is_connected.return_value = True
        fake_w3.eth.chain_id = 56
        fake_w3.eth.get_balance.return_value = 10**18  # 1 BNB
        fake_w3.eth.gas_price = 5 * 10**9
        fake_w3.eth.get_transaction_count.return_value = 1
        fake_w3.to_wei.side_effect = lambda v, u: int(Decimal(str(v)) * Decimal(10**18))
        fake_w3.from_wei.side_effect = lambda v, u: Decimal(v) / Decimal(10**18)
        fake_w3.eth.account.sign_transaction.return_value = mock.MagicMock(
            raw_transaction=b"\x00" * 32)
        fake_w3.eth.send_raw_transaction.side_effect = ValueError(
            "replacement transaction underpriced")
        # Web3.to_checksum_address is a classmethod-style helper - we want the
        # real implementation, not a mock, so we patch the constructor only.

        with mock.patch.object(styx_register, "send_bnb_deposit",
                               wraps=styx_register.send_bnb_deposit):
            with mock.patch("web3.Web3.HTTPProvider"), \
                 mock.patch("web3.Web3", return_value=fake_w3) as web3_cls:
                web3_cls.to_checksum_address = lambda addr: addr  # passthrough
                # Speed up: zero out the retry backoff sleep
                with mock.patch("styx_register.time.sleep"):
                    with self.assertRaises(RuntimeError) as ctx:
                        styx_register.send_bnb_deposit(
                            to_address=GOOD_ADDR,
                            amount_bnb=0.001,
                            max_retries=3,
                        )
        # We expect "failed after 3 attempts" in the final error message
        msg = str(ctx.exception)
        self.assertIn("failed after 3 attempts", msg)
        # And send_raw_transaction must have been called exactly 3 times
        self.assertEqual(fake_w3.eth.send_raw_transaction.call_count, 3)


if __name__ == "__main__":
    unittest.main()
