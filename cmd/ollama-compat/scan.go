package main

import (
	"github.com/spf13/cobra"

	"github.com/shra26/ollama-compatibility-checker/internal/output"
)

func newScanCmd(detect detectFunc) *cobra.Command {
	return &cobra.Command{
		Use:   "scan",
		Short: "Report detected hardware",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			h := detect(cmd.Context())
			if asJSON, _ := cmd.Flags().GetBool("json"); asJSON {
				return output.WriteJSON(cmd.OutOrStdout(), output.ScanDoc(h))
			}
			output.RenderScan(cmd.OutOrStdout(), h, useColor(cmd))
			return nil
		},
	}
}
